
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import tarfile

main_dir='final_glioma2/final_glioma'
learning_rate = 0.01
display_step = 10

#Cropped image dimensions

dim_st_width=0
dim_st_width=int(dim_st_width)

dim_st_ht=0
dim_st_ht=int(dim_st_ht)

dim_end_width=200
dim_end_width=int(dim_end_width)

dim_end_ht=200
dim_end_ht=int(dim_end_ht)

image_width= int(dim_end_width - dim_st_width)
image_ht= int(dim_end_ht- dim_st_ht)

#Dimensions of the densely connected layer of CNN

wd1_dim_width=image_width/8
wd1_dim_width=int(wd1_dim_width)

wd1_dim_ht=image_ht/8
wd1_dim_ht=int(wd1_dim_ht)

# Network Parameters
n_input = image_width*image_ht
n_classes = 4 
n_hidden=64
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, image_width ,image_ht])
y = tf.placeholder(tf.float32, [1, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)



# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 32])),
    # fully connected, 150*150*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([wd1_dim_width*wd1_dim_ht*32, 512])),

    'out': tf.Variable(tf.random_normal([512, n_classes]))
    
}


biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

weights_lstm = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases_lstm = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

print("starting")
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout=0.75):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1,image_width,image_ht, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    conv3= conv2d(conv2,weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    #fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    fc1=tf.reshape(fc1,shape=[1,100,512])
    return fc1


def lstm(fc,weights,biases):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, fc, dtype=tf.float32)

    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs=tf.gather(outputs,99)
    outputs= tf.reshape(outputs,[-1, n_hidden])
    print(outputs.shape)
    return tf.matmul(outputs, weights['out']) + biases['out']

def get_next_batch(step):

    batch=[]
    mri=os.listdir(main_dir)[step]
    for image in os.listdir(main_dir+'/'+mri):
        if image== main_dir+'/'+mri:
            continue
        img=Image.open(main_dir+'/'+mri+'/'+image)
        img=img.convert('L')
        area = (dim_st_width, dim_st_ht, dim_end_width, dim_end_ht)
        cropped_img = img.crop(area)
        img=np.asarray(cropped_img)
        batch.append(img)

    batch=np.array(batch)
    a=np.zeros(shape=[100,image_width,image_ht])
    a[:batch.shape[0],:batch.shape[1],:batch.shape[2]] = batch

    grade= mri.split('--')[1].split('--')[0]
    if   grade == 'grade 3 positive':
      true_label = 0
             
    elif grade == 'grade 3 wildtype':
      true_label = 1
            
    elif grade == 'grade 4 positive':
      true_label = 2
            
    elif grade =='grade 4 wildtype':
      true_label = 3

    b=np.zeros(shape=[1,4])
    b[0][true_label]=1

    return list(a),list(b)

print("starting")

# Construct model
fc = conv_net(x, weights, biases)
pred= lstm(fc, weights_lstm, biases_lstm )

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()

no_of_mris=455
step=0
with tf.Session() as sess:
    sess.run(init)
    print("getting batch ",step)
    batch_x,batch_y= get_next_batch(step)

    while step < no_of_mris:
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        #pred1=sess.run(pred, feed_dict={x: batch_x, y: batch_y})
        #print(batch_y,pred1)
        if step % display_step == 0:
            print("discard")
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step) + ", Minibatch Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

        step += 1
        break
print("Optimization Finished!")

