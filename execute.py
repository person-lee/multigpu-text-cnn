# coding=utf-8
import time
import logging
import tensorflow as tf
import codecs

from collections import defaultdict
import numpy as np

from cnn import CNN
from data_helper import load_data, load_embedding, batch_iter, create_valid, build_vocab, build_label
from utils import convert_map_to_array, tower_loss, cal_predictions, multigpu_cal

#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("all_file", "../data/baseLine/all.txt", "all corpus file")
tf.flags.DEFINE_string("train_file", "../data/baseLine/train.txt", "train corpus file")
tf.flags.DEFINE_string("test_file", "../data/baseLine/test.txt", "test corpus file")
tf.flags.DEFINE_string("word_file", "../data/baseLine/words.txt", "test corpus file")
tf.flags.DEFINE_string("embedding_file", "../data/baseLine/vectors.txt", "vector file")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5", "filter size of cnn")
tf.flags.DEFINE_integer("embedding_size", 150, "embedding size")
tf.flags.DEFINE_integer("sequence_len", 80, "embedding size")
tf.flags.DEFINE_integer("num_filters", 128, "the number of filter in every layer")
tf.flags.DEFINE_float("dropout", 0.5, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 256, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 70, "epoches")
tf.flags.DEFINE_integer("evaluate_every", 2000, "run evaluation")
tf.flags.DEFINE_integer("l2_reg_lambda", 0.0, "l2 regulation")
tf.flags.DEFINE_string("device_num", "1,2,3,4", "How many Devices(cpu/gpu) to use.")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.4, "use memory rate")

FLAGS = tf.flags.FLAGS
gpu_list = FLAGS.device_num.split(",")
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger ---------------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log", mode="w")
fh.setLevel(logging.INFO)
logger.addHandler(fh)
#----------------------------- define a logger end -----------------------------------

#------------------------------------load data -------------------------------

label2idx, idx2label = build_label(FLAGS.all_file)
word2idx, idx2word = build_vocab(FLAGS.word_file)
total_y, total_x = load_data(FLAGS.train_file, word2idx, FLAGS.sequence_len, label2idx)
logger.info("load train data finish")
#train_data, valid_data = create_valid(zip(total_x, total_y))
#train_x, train_y = zip(*train_data)
#valid_x, valid_y = zip(*valid_data)
train_x, train_y = total_x, total_y
num_classes = len(label2idx)
embedding = load_embedding(FLAGS.embedding_size, filename=FLAGS.embedding_file)
test_y, test_x = load_data(FLAGS.test_file, word2idx, FLAGS.sequence_len, label2idx)
logger.info("load test data finish")
#----------------------------------- load data end ----------------------

#----------------------------------- cal filter_size --------------------------------------
filter_sizes = [int(filter_size.strip()) for filter_size in FLAGS.filter_sizes.strip().split(",")]
#----------------------------------- cal filter_size end ----------------------------------

#----------------------------------- step -------------------------------------------------
def run_step(sess, cnn, batch_x, batch_y, dropout=1., is_optimizer=True):
    start_time = time.time()
    feed_dict = {
        input_x:batch_x,
        input_y:batch_y, 
        cnn.dropout:dropout
    }

    step, cur_loss, cur_acc, predicts, _ = sess.run([global_step, avg_loss, avg_acc, predictions , train_op], feed_dict)

    elapsed_time = time.time() - start_time
    return step, cur_loss, cur_acc, predicts, elapsed_time

def valid_step(sess, cnn, batch_x, batch_y, dropout=1., is_optimizer=True):
    start_time = time.time()
    feed_dict = {
        input_x:batch_x,
        input_y:batch_y, 
        cnn.dropout:dropout
    }

    step, cur_loss, cur_acc, predicts = sess.run([global_step, avg_loss, avg_acc, predictions ], feed_dict)

    elapsed_time = time.time() - start_time
    return step, cur_loss, cur_acc, predicts, elapsed_time

#----------------------------------- step end ---------------------------------------------

#----------------------------------- validate model --------------------------------------
def validate_model(sess, cnn, valid_x, valid_y):
    start_time = time.time()
    batches = batch_iter(zip(valid_x, valid_y), FLAGS.batch_size, shuffle=False)
    total_loss, total_acc, total_elapsed_time = 0, 0, 0
    idx = 0
    pred_labels = list()
    act_labels = list()
    for batch in batches:
        batch_x, batch_y = zip(*batch)
        step, cur_loss, cur_acc, predicts, elapsed_time = valid_step(sess, cnn, batch_x, batch_y, is_optimizer=False)
        total_loss += cur_loss
        total_acc += cur_acc
        total_elapsed_time += elapsed_time
        idx += 1
        pred_labels.extend(predicts)
        act_labels.extend(batch_y)

    aver_loss = 1. * total_loss / idx
    aver_acc = 1. * total_acc / idx
    aver_elapsed_time = 1. * total_elapsed_time / idx
    validate_elapsed_time = time.time() - start_time
    logger.info("validation loss:%s, acc:%s, %6.7f secs/batch_size, total elapsed time: %6.7f"%(aver_loss, aver_acc, aver_elapsed_time, validate_elapsed_time))
    return pred_labels, act_labels

def test_model(sess, cnn, valid_x, valid_y):
    pred_labels, act_labels = validate_model(sess, cnn, valid_x, valid_y)
    ori_quests = [line.strip() for line in codecs.open(FLAGS.test_file, "r", "utf-8").readlines()]
    #if len(ori_quests) != len(pred_labels):
    #    logger.error("error, length is not same")
    #else:
    with codecs.open("testRet.txt", "w", "utf-8") as wf:
        for idx in np.arange(len(act_labels)):
            wf.write(ori_quests[idx] + " " + idx2label[pred_labels[idx]] + "\n")
    logger.info("save predict result success")
    
#----------------------------------- validate model end ----------------------------------

#----------------------------------- execute train ---------------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            cnn = CNN(FLAGS.batch_size, FLAGS.sequence_len, embedding, FLAGS.embedding_size, filter_sizes, FLAGS.num_filters, num_classes, l2_reg_lambda=FLAGS.l2_reg_lambda)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-4)

            input_x = tf.placeholder(tf.int32, [None, FLAGS.sequence_len])
            input_y = tf.placeholder(tf.int32, [None,])

            avg_loss, avg_acc, grads_and_vars, predictions = multigpu_cal(cnn, optimizer, input_x, input_y, gpu_list)

            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            sess.run(tf.initialize_all_variables())

            # load batch data
            for epoch in range(FLAGS.epoches):
                batches = batch_iter(zip(train_x, train_y), FLAGS.batch_size)
                for batch in batches:
                    batch_x, batch_y = zip(*batch)
                    step, cur_loss, cur_acc, predicts, elapsed_time = run_step(sess, cnn, batch_x, batch_y, FLAGS.dropout)
                    logger.info("%s steps, loss:%s, acc:%s, %6.7f secs/batch_size"%(step, cur_loss, cur_acc, elapsed_time))
                    cur_step = tf.train.global_step(sess, global_step)

                    #if cur_step != 0 and cur_step % FLAGS.evaluate_every == 0:
                    #    logger.info("************** start to evaluate model *****************")
                    #    validate_model(sess, cnn, valid_x, valid_y)

            # test model
            logger.info("********************* start to test model ****************************")
            test_model(sess, cnn, test_x, test_y)
            logger.info("********************* end to test model ****************************")
#----------------------------------- execute train end -----------------------------------
