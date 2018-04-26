"""
Train model specified in class static var: TrainConfig.model_name.

Key Features:
  1. Saves key operations and variables for viewing in TensorBoard
  2. Training control:
    a) Learning rate decreases based on validation accuracy trend
    b) Training terminates based on validation accuracy trend
  4. Basic user interface to:
    a) Name training runs / directories
    b) Saves copy of .py files in training result directory
    c) Resume training from checkpoint
"""
from vgg_16 import *
# from logistic_regression import *
# from single_layer_nn import *
from metrics import *
from losses import *
from input_pipe import *
from datetime import datetime
import numpy as np
import os
import shutil
import glob


class TrainConfig(object):
  """Training configuration"""
  batch_size = 64
  num_epochs = 50
  summary_interval = 250
  eval_interval = 2000  # must be integer multiple of summary_interval
  lr = 0.01  # learning rate
  reg = 5e-4  # regularization
  momentum = 0.9
  dropout_keep_prob = 0.5
  model_name = 'vgg_16'  # choose model
  model = staticmethod(globals()[model_name])  # gets model by name


class TrainControl(object):
  """Basic training control

  Decreases learning rate (lr), terminates training after 3 lr decreases

  Track validation accuracy, decrease lr by 1/5th when:
    1. validation accuracy worsens
    2. less than 0.2% absolute improvement last 3 iterations
  """
  def __init__(self, lr):
    self.val_accs = []
    self.lr = lr
    self.num_lr_updates = 0
    self.lr_factor = 1/5

  def add_val_acc(self, loss):
    self.val_accs.append(loss)

  def update_lr(self, sess):
    if len(self.val_accs) < 3:
      return
    decrease = False
    # decrease LR if validation acc worsens
    if self.val_accs[-1] < max(self.val_accs):
      decrease = True
    avg_2 = (self.val_accs[-2] + self.val_accs[-3]) / 2
    # decrease LR if validation accuracy doesn't improve by 0.2% (absolute)
    if abs(self.val_accs[-1] - avg_2) < 0.002:
      decrease = True
    if decrease:
      old_lr = sess.run(self.lr)
      self.lr.load(old_lr * self.lr_factor)
      self.num_lr_updates += 1
      print('*** New learning rate: {}'.format(old_lr * self.lr_factor))

  def done(self):
    if self.num_lr_updates > 3:  # terminate training after 3 lr decreases
      return True
    else:
      return False


def optimizer(loss, config):
  """Add training operation, global_step and learning rate variable to Graph

  Args:
    loss: model loss tensor
    config: training configuration object

  Returns:
    (train_op, global_step, lr)
  """
  lr = tf.Variable(config.lr, trainable=False, dtype=tf.float32)
  global_step = tf.Variable(0, trainable=False, name='global_step')
  optim = tf.train.MomentumOptimizer(lr, config.momentum,
                                     use_nesterov=True)
  train_op = optim.minimize(loss, global_step=global_step)

  return train_op, global_step, lr


def get_logdir():
  """Return unique logdir based on datetime"""
  now = datetime.utcnow().strftime("%m%d%H%M%S")
  logdir = "run-{}/".format(now)

  return logdir


def model(mode, config):
  """Pull it all together: input queue, inference model and loss functions

  Args:
    mode: 'train' or 'val'
    config: model configuration object

  Returns:
    loss and accuracy tensors
  """
  # preprocess images on cpu - send to gpu as uint8 for speed
  with tf.device('/cpu:0'):
    imgs, labels = batch_q(mode, config)

  logits = config.model(imgs, config)
  softmax_ce_loss(logits, labels)
  acc = accuracy(logits, labels)
  total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')
  total_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                         name='total_loss') * config.reg
  for l2 in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
    # add l2 loss histograms to TensorBoard and cleanup var names
    name = 'l2_loss_' + l2.name.split('/')[0]
    tf.summary.histogram(name, l2)

  return total_loss, acc


def evaluate(ckpt):
  """Load checkpoint and run on validation set"""
  config = TrainConfig()
  config.dropout_keep_prob = 1.0  # disable dropout for validation
  config.num_epochs = 1
  accs, losses = [], []

  with tf.Graph().as_default():
    loss, acc = model('val', config)
    saver = tf.train.Saver()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    with tf.Session() as sess:
      init.run()
      saver.restore(sess, ckpt)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        while not coord.should_stop():
          step_loss, step_acc = sess.run([loss, acc])
          accs.append(step_acc)
          losses.append(step_loss)
      except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
      finally:
        coord.request_stop()
        coord.join(threads)
  mean_loss, mean_acc = np.mean(losses), np.mean(accs)
  print('Validation. Loss: {:.3f}, Accuracy: {:.4f}'.
        format(mean_loss, mean_acc))

  return mean_loss, mean_acc


def options(config):
  """Get user input on training options"""
  q = input('Enter a short configuration name [default = "default"]: ')
  if len(q) == 0:
    q = 'default'
  config.config_name = q
  # tensorboard and checkpoint log directory names
  ckpt_path = 'checkpoints/' + config.model_name + '/' + config.config_name
  tflog_path = ('tf_logs/' + config.model_name + '/' +
                config.config_name + '/' + get_logdir())
  checkpoint = None
  # TODO: spaghetti mess, clean up:
  if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)
    filenames = glob.glob('*.py')
    for filename in filenames:
      shutil.copy(filename, ckpt_path)
    return False, ckpt_path, tflog_path, checkpoint
  else:
    filenames = glob.glob('*.py')
    for filename in filenames:
      shutil.copy(filename, ckpt_path)
    while True:
      q1 = input('Continue previous training? [Y/n]: ')
      if len(q1) == 0 or q1 == 'n' or q1 == 'Y':
        break
    if q1 == 'n':
      return False, ckpt_path, tflog_path, checkpoint
    else:
      q2 = input('Enter checkpoint name [defaults to most recent]: ')
      if len(q2) == 0:
        checkpoint = tf.train.latest_checkpoint(ckpt_path)
      else:
        checkpoint = ckpt_path + '/' + q2
      return True, ckpt_path, tflog_path, checkpoint


def train():
  """Build Graph, launch session and train."""

  config = TrainConfig()
  continue_train, ckpt_path, tflog_path, checkpoint = options(config)
  g = tf.Graph()
  with g.as_default():
    loss, acc = model('train', config)
    train_op, g_step, lr = optimizer(loss, config)
    controller = TrainControl(lr)
    # put variables in graph to hold validation acc and loss for TensorBoard viewing
    val_acc = tf.Variable(0.0, trainable=False)
    val_loss = tf.Variable(0.0, trainable=False)
    tf.summary.scalar('val_loss', val_loss)
    tf.summary.scalar('val_accuracy', val_acc)
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    # histograms of all variables to TensorBoard
    [tf.summary.histogram(v.name.replace(':', '_'), v)
     for v in tf.trainable_variables()]
    # next line only needed for batch normalization (updates beta and gamma)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    summ = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=1)
    writer = tf.summary.FileWriter(tflog_path, g)
    with tf.Session() as sess:
      init.run()
      if continue_train:
        saver.restore(sess, checkpoint)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        losses, accs = [], []  # hold running averages for test loss/acc
        while not coord.should_stop():
          step_loss, _, step, step_acc, __ = sess.run([loss, train_op,
                                                       g_step, acc, extra_update_ops])
          losses.append(step_loss)
          accs.append(step_acc)
          if step % config.eval_interval == 0:
            ckpt = saver.save(sess, ckpt_path + '/model', step)
            mean_loss, mean_acc = evaluate(ckpt)
            val_acc.load(mean_acc)
            val_loss.load(mean_loss)
            controller.add_val_acc(mean_acc)
            controller.update_lr(sess)
            if controller.done():
              break
          if step % config.summary_interval == 0:
            writer.add_summary(sess.run(summ), step)
            print('Iteration: {}, Loss: {:.3f}, Accuracy: {:.4f}'.
                  format(step, np.mean(losses), np.mean(accs)))
            losses, accs = [], []
      except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
      finally:
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
  train()
