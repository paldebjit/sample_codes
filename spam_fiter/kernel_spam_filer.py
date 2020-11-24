import os
import numpy as np
import heterocl as hcl
from config import *
from lut import *

def top_spam_filer(dtype=hcl.Int(), target=None):

    hcl.init(dtype)

    data = hcl.placeholder((NUM_FEATURES * NUM_TRAINING,), "data")
    label = hcl.placeholder((NUM_TRAINING,), "label")
    theta = hcl.placeholder((NUM_FEATURES,), "theta")
    
    # Call: dotProduct(theta_local, training_instance, dot)
    def dotProduct(param, feature, dot):

        with hcl.for_(0, NUM_FEATURES / PAR_FACTOR, name="DOT") as i:
            with hcl.for_(0, PAR_FACTOR, name="DOT_INNER") as j:
                term = param[i * PAR_FACTOR + j] * feature[i * PAR_FACTOR + j]
                dot += term

    def useLUT(in_):
        
        with hcl.if_(in_ < 0):
            in_ = (int)(-1 * in_)
            index = LUT_SIZE - (in_) << (LUTIN_TWIDTH - LUTIN_IWIDTH)
        with hcl.else_():
            index = (int)(in_) << (LUTIN_TWIDTH - LUTIN_IWIDTH)

        return lut[index]
    
    # Call: Sigmoid(dot, prob)
    def Sigmoid(exponent, prob):
        
        with hcl.if_(exponent > 4):
            prob.v = 1.0
        with hcl.elif_(exponent < -4):
            prob.v = 0.0
        with hcl.else_():
            prob.v = useLUT(exponent)

    # Call: computeGradient(gradient, training_instance, scale)
    def computeGradient(grad, feature, scale):
    
        with hcl.for_(0, NUM_FEATURES / PAR_FACTOR, name="GRAD") as i:
            with hcl.for_(0, PAR_FACTOR, name="GRAD_INNER") as j:
                grad[i * PAR_FACTOR + j] = (scale.v * feature[i * PAR_FACTOR + j])

    # Call: updateParameter(theta_local, gradient, step)
    def updateParameter(param, grad, scale):

        with hcl.for_(0, NUM_FEATURES / PAR_FACTOR, name="UPDATE") as i:
            with hcl.for_(0, PAR_FACTOR, name="UPDATE_INNER") as j:
                param[i * PAR_FACTOR + j] += (scale.v * grad[i * PAR_FACTOR + j])
    
    def compute(theta_local, label_local, training_instance, training_id):
        
        gradient = hcl.compute((NUM_FEATURES,), lambda x:0, 
                               name="gradient",
                               dtype=dtype)
        step = hcl.scalar(-1 * STEP_SIZE)

        dot = 0.0
        dotProduct(theta_local, training_instance, dot)

        prob = hcl.scalar(0.0)
        Sigmoid(dot, prob)

        training_label = hcl.scalar(label_local[training_id])
        
        scale = hcl.scalar(prob.v - training_label.v)
        computeGradient(gradient, training_instance, scale)
        
        updateParameter(theta_local, gradient, step)

    def read_data(data, training_instance, training_id):

        with hcl.for_(0, NUM_FEATURES, name="i") as i:
            training_instance[i] = data[training_id * NUM_FEATURES + i]

    def training(theta_local, label_local, data, training_id):

        training_instance = hcl.compute((NUM_FEATURES,), lambda x:0, 
                                        name="training_instance",
                                        dtype=dtype)

        read_data(data, training_instance, training_id)
        compute(theta_local, label_local, training_instance, training_id)

    def epoch(theta_local, label_local, data):
        
        # Repetatively run epoch function for # NUM_TRAINING
        hcl.mutate((NUM_TRAINING,), lambda m: training(theta_local, label_local, data, m), "TRAINING_INST")

    def kernel_spam_filter(data, label, theta):

        theta_local = hcl.compute((NUM_FEATURES,), lambda x: 0, name="theta_local")
        label_local = hcl.compute((NUM_TRAINING,), lambda x: 0, name="label_local")

        with hcl.for_(0, NUM_FEATURES, name="i") as i:
            theta_local[i] = theta[i]
      
        with hcl.for_(0, NUM_TRAINING, name="i") as i:
            label_local[i] = label[i]

        # Repetatively run epoch function for # NUM_EPOCHS
        hcl.mutate((NUM_EPOCHS,), lambda m: epoch(theta_local, label_local, data), "EPOCH")

        with hcl.for_(0, NUM_FEATURES, name="i") as i:
            theta[i] = theta_local[i]

    s = hcl.create_schedule([data, label, theta], kernel_spam_filter)
    return hcl.build(s, target=target)


def codegen():
    os.makedirs('device', exist_ok=True)
    targets = ['vhls', 'aocl']
    
    for target in targets:
        f = top_spam_filer(dtype=hcl.Float(), target=target)
        fp = open('device/kernel_' + target + '.cl', 'w')
        fp.write(f)
        fp.close()

def test():
    
    Ddtype = hcl.Float()
    Ldtype = hcl.Float()
    Tdtype = hcl.Float()
    
    np_data = hcl.cast_np(np.loadtxt("data/shuffledfeats.dat"), Ddtype)
    np_label = hcl.cast_np(np.loadtxt("data/shuffledlabels.dat"), Ldtype)
    np_theta = hcl.cast_np(np.zeros((NUM_FEATURES,), dtype=float), Tdtype)

    np_train_data = np_data[:NUM_FEATURES * NUM_TRAINING]
    np_train_label = np_label[:NUM_TRAINING]

    np_test_data = np_data[NUM_FEATURES * NUM_TRAINING:]
    np_test_label = np_label[NUM_TRAINING:]

    hcl_data = hcl.asarray(np_train_data, dtype=Ddtype)
    hcl_label = hcl.asarray(np_label, dtype=Ldtype)
    hcl_theta = hcl.asarray(np_theta, dtype=Tdtype)
    
    dtype = hcl.Float()
    f = top_spam_filer(dtype)

    f(hcl_data, hcl_label, hcl_theta)

    theta_out = hcl_theta.asnumpy()
    
    #print(theta_out)

    error = 0.0
    for i in range(NUM_TESTING):
        data = np_test_data[i * NUM_FEATURES : (i + 1) * NUM_FEATURES]
        dot = 0.0
        for j in range(NUM_FEATURES):
            dot += data[j] * theta_out[j]
        
        result = 1.0 if dot > 0 else 0.0

        #print("Result is: %f and np_test_label[%d] is: %f.\n" % (result, i, np_test_label[i]))

        if result != np_test_label[i]:
            error += 1.0

    print("The average error is: %f.\n" % (error / NUM_TESTING))

if __name__ == "__main__":
    test()
    codegen()
