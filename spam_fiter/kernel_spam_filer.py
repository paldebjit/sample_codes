import heterocl as hcl
from config import *

def top_spam_filer(dtype=hcl.Int(), target=None):

    hcl.init(dtype)

    data = hcl.placeholder((NUM_FEATURES * NUM_TRAINING,), "data")
    label = hcl.placeholder((NUM_TRAINING,), "label")
    theta = hcl.placeholder((NUM_FEATURES,), "theta")

    def compute(theta_local, label_local, training_instance, training_id):
        
        gradient = hcl.compute((NUM_FEATURES,), lambda x:0, name="gradient")
        step = hcl.scalar(-1 * STEP_SIZE)

        dot = hcl.scalar(0.0)
        dotProduct(theta_local, training_instance, dot)

        prob = hcl.scalar(0.0)
        Sigmoid(dot, prob)

        training_label = hcl.scalar(label_local[training_id])
        
        scale = hcl.scalar(prob - training_label)
        computeGradient(gradient, training_instance, scale)
        
        updateParameter(theta_local, gradient, step)

    def read_data(data, training_instance, training_id):

        with hcl.for_(0, NUM_FEATURES, name="i") as i:
            training_instance[i] = data[training_id * NUM_FEATURES + i]

    def training(theta_local, label_local, data, training_id):

        training_instance = hcl.compute((NUM_FEATURES,), lambda x:0, name="training_instance")

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
