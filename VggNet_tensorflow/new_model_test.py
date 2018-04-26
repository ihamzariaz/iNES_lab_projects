######################Testing#################################
from VGG16_model import vgg16
from scipy.misc import imread, imresize
import tensorflow as tf
num_classes=4;
names = ['cats','dogs','horses','humans']
# build the graph
graph = tf.Graph()
with graph.as_default():
    input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
    # zero mean of input
    #mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[1, 1, 1, 3])
    output = vgg16(input_maps, num_classes, isTrain=True, keep_prob=0.6)
    softmax = tf.nn.softmax(output)
    # Finds values and indices of the k largest entries
    values, indices = tf.nn.top_k(softmax)

# read sample image
img = imread('test7.png', mode='RGB')
img = imresize(img, [224, 224])

# run the graph
with tf.Session(graph=graph) as sess:
    # restore model parameters
    saver = tf.train.Saver()
    print('Restoring VGG16 model parameters ...')
#    saver.restore(sess, 'model.ckpt.meta') 
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    sess.run(tf.initialize_all_variables())
#    print('Restoring VGG16 model parameters ...')
#    saver.restore(sess, 'VGG16_modelParams.tensorflow')
    # testing on the sample image
    [prob, ind, out] = sess.run([values, indices, output], feed_dict={input_maps: [img]})
    prob = prob[0]
    ind = ind[0]
    print('\nClassification Result:')
    print('\tCategory Name: %s \n\tProbability: %.2f%%\n' % (names[ind[0]], prob[0]*100))
    sess.close()
