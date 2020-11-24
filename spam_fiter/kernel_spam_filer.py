import os
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

    s = hcl.create_schedule([data, label, theta], kernel_spam_filter)
    return hcl.build(s, target=target)


os.makedirs('device', exist_ok=True)
targets = ['vhls', 'aocl']

for target in targets:
    f = top_spam_filer(dtype=hcl.Float(), target=target)
    fp = open('device/kernel_' + target + '.cl', 'w')
    fp.write(f)
    fp.close()


