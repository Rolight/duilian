import json
import jieba
import copy
import pickle
import os
import random
import numpy as np
import tensorflow as tf


def load_pretrained_cn_word2vec():
    import gensim
    print('loading pretrained word2vec from hard disk...')
    cn_word2vec = gensim.models.KeyedVectors.load_word2vec_format(
        './cn_word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin',
        binary=True
    )
    return cn_word2vec


def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fff':
        return True
    else:
        return False


class DuiLianUtils:
    # This model save the preprocessed training data

    UNK = '<UNK>'
    EOS = '<EOS>'
    SOS = '<SOS>'
    PUNC = '<PUNC>'

    def do_preprocess(self):
        pretrained_model = load_pretrained_cn_word2vec()
        self.preprocess_word2vec(pretrained_model)
        self.preprocess_train_data()

    def preprocess_word2vec(self, model):
        print('preprocessing word2vec model, fetch some information.')

        self.word_list = copy.copy(model.index2word)

        # because load all words will cost more than 3GB memory,
        # so we only use single word
        self.single_word_mask = [idx for idx, word in enumerate(
            self.word_list) if len(word) == 1]
        self.word_list = [word for word in self.word_list if len(word) == 1]

        self.word2index = dict(zip(self.word_list, range(len(self.word_list))))
        self.wordvectors = np.array(model.vectors)
        # get wordvectors
        self.wordvector_dim = self.wordvectors.shape[1]
        self.wordvectors = self.wordvectors[self.single_word_mask, :]

        self.extra_tags = [self.UNK, self.EOS, self.SOS, self.PUNC]

        # Add extra tag
        for tag in self.extra_tags:
            self.word2index[tag] = len(self.word_list)
            self.word_list.append(tag)
            self.wordvectors = np.concatenate(
                (self.wordvectors, np.random.uniform(
                    low=-1, high=1,
                    size=(1, self.wordvector_dim)))
            )

        self.word_set = set(self.word_list)
        self.word_amount = len(self.word_list)

    def preprocess_train_data(self):
        print('loading and processing train data')
        with open('duilian.json') as f:
            data = json.loads(f.read())
        self.raw_data = []
        self.max_length = 0
        self.inputs = []
        self.labels = []
        self.inputs_length = []
        self.words_seq = []

        # Swap the left roll and right roll to increase train data
        for d in data:
            self.raw_data.append([d[0], d[1]])
            self.raw_data.append([d[1], d[0]])

        self.input_size = len(self.raw_data)

        for d in self.raw_data:
            left_roll_word_seq, left_roll_index_seq = self.convert_cn_sentence(
                d[0], reverse=True)
            right_roll_word_seq, right_roll_index_seq = self.convert_cn_sentence(
                d[1], reverse=False)
            self.words_seq.append((left_roll_word_seq, right_roll_word_seq))
            self.inputs_length.append(len(left_roll_index_seq))
            self.inputs.append(left_roll_index_seq)
            self.labels.append(right_roll_index_seq)
        self.max_length = max(self.inputs_length) + 5

        input_mat = np.zeros((self.input_size, self.max_length))
        decode_inputs_mat = np.zeros(
            (self.input_size, self.max_length), dtype=np.int32)
        decode_outputs_mat = np.zeros(
            (self.input_size, self.max_length), dtype=np.int32)

        for idx in range(self.input_size):
            input_seq = self.inputs[idx]
            label_seq = self.labels[idx]
            input_mat[idx, 0:len(input_seq)] = input_seq
            decode_inputs_mat[idx, 1:len(label_seq)+1] = label_seq
            decode_inputs_mat[idx, 0] = self.word2index[self.SOS]
            decode_outputs_mat[idx, 0:len(label_seq)] = label_seq
            decode_outputs_mat[idx, len(label_seq)] = self.word2index[self.EOS]

        self.inputs = input_mat
        self.decode_inputs = decode_inputs_mat
        self.decode_outputs = decode_outputs_mat
        self.inputs_length = np.array(self.inputs_length, dtype=np.int32)
        self.outputs_length = self.inputs_length + 1

    def convert_cn_sentence(self, sentence, reverse=False):
        """
             Convert sentence in Chinese to word index sequence.
             Using jieba to split the sentence into words sequence,
             if words not in pretrained word2vec, split words to single Chinese characters
             or using <UNK> label
        """
        cutted_sentence = list(jieba.cut(sentence))
        words_seq = []
        for word in cutted_sentence:
            if word in self.word_set:
                words_seq.append(word)
            else:
                for ch in word:
                    if ch in self.word_set:
                        words_seq.append(ch)
                    elif is_chinese(ch):
                        words_seq.append(self.UNK)
                    else:
                        words_seq.append(self.PUNC)
        index_seq = []
        for word in words_seq:
            index_seq.append(self.word2index[word])
        # input sentence may need to reverse
        if reverse:
            words_seq.reverse()
            index_seq.reverse()
        return words_seq, index_seq


class DuiLianModel:

    def __init__(self, model_name='duilian', batch_size=64, rnn_layer_num=3,
                 rnn_layer_dim=128, learning_rate=0.001,
                 l2_reg=0.001, use_pretrained_embedding=True):
        # Some constants
        self.model_name = model_name
        self.rnn_layer_num = rnn_layer_num
        self.rnn_layer_dim = rnn_layer_dim
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.max_grad_norm = 5.0
        self.batch_size = batch_size
        self.utils_path = 'duilian_utils.pkl'
        self.beam_width = 30
        self.utils = self.init_utils()

        # load attr from utils
        self.max_length = self.utils.max_length
        self.inputs = self.utils.inputs
        self.word_amount = self.utils.word_amount
        self.inputs_length = self.utils.inputs_length
        self.outputs_length = self.utils.outputs_length
        self.decode_inputs = self.utils.decode_inputs
        self.decode_outputs = self.utils.decode_outputs
        self.word2index = self.utils.word2index
        self.word_list = self.utils.word_list
        self.n_classes = self.word_amount

        if use_pretrained_embedding:
            self.wordvectors = self.utils.wordvectors
        else:
            self.wordvectors = np.random.uniform(
                low=-1, high=1, size=self.utils.wordvectors.shape)

        self.save_path = './%s_saved/duiliam_model.ckpt' % self.model_name
        self.save_folder = './%s_saved' % self.model_name
        self.build()

    def init_utils(self):
        if not os.path.exists(self.utils_path):
            print('The utils pickle file doesn\'t exist. Create it first.')
            utils = DuiLianUtils()
            utils.do_preprocess()
            print('Saving utils object to disk...')
            with open(self.utils_path, 'wb') as f:
                f.write(pickle.dumps(utils))
            return utils
        else:
            print('Loading utils pickle file.')
            with open(self.utils_path, 'rb') as f:
                utils = pickle.loads(f.read())
            return utils

    def build(self):
        self.add_placeholder()
        with tf.variable_scope('duilian', reuse=tf.AUTO_REUSE):
            """
            self.pretrained_embeddings = tf.Variable(
                np.random.randn(*self.wordvectors.shape), dtype=tf.float32
            )
            """
            self.pretrained_embeddings = tf.Variable(
                self.wordvectors, dtype=tf.float32)
            self.logits = self.add_train_prediction_op()
            self.prediction = self.add_test_prediction_op()
            self.loss = self.add_loss_op(self.logits)
            self.train_op = self.add_train_op(self.loss)
            self.sess = self.init_Session()
            if os.path.exists(self.save_folder):
                try:
                    self.saver = tf.train.Saver()
                    self.saver.restore(self.sess, self.save_path)
                    print('Loaded saved model from disk.')
                except Exception:
                    print('Load model error!')
            else:
                print('Cannot find saved model.')
                self.init_variables()

    def init_Session(self):
        sess = tf.Session()
        return sess

    def init_variables(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_placeholder(self):
        self.inputs_placeholder = tf.placeholder(
            shape=(None, self.max_length),
            dtype=tf.int32
        )
        self.input_length_placeholder = tf.placeholder(
            shape=(None, ),
            dtype=tf.int32
        )
        self.decode_inputs_placeholder = tf.placeholder(
            shape=(None, self.max_length),
            dtype=tf.int32
        )
        self.decode_outputs_placeholder = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32
        )
        self.decode_length_placeholder = tf.placeholder(
            shape=(None,),
            dtype=tf.int32
        )
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)

    def create_feed_dict(self, inputs_batch, input_length, decode_length,
                         decode_inputs, decode_outputs, dropout_rate):
        feed_dict = {
            self.inputs_placeholder: inputs_batch,
            self.input_length_placeholder: input_length,
            self.dropout_placeholder: dropout_rate,
        }
        if decode_length is not None:
            decode_outputs = decode_outputs[:, :np.max(decode_length)]
            feed_dict[self.decode_inputs_placeholder] = decode_inputs
            feed_dict[self.decode_outputs_placeholder] = decode_outputs
            feed_dict[self.decode_length_placeholder] = decode_length
        return feed_dict

    def add_embedding(self, indices):
        embeddings = tf.nn.embedding_lookup(
            self.pretrained_embeddings,
            indices
        )
        return embeddings

    def add_stacked_gru_cell(self):
        # initializer = tf.contrib.layers.xavier_initializer()
        dropout_rate = self.dropout_placeholder
        cells = [
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(
                    num_units=self.rnn_layer_dim,
                    # kernel_initializer=initializer
                ),
                output_keep_prob=dropout_rate
            ) for i in range(self.rnn_layer_num)]
        stacked_gru = tf.contrib.rnn.MultiRNNCell(cells)
        return stacked_gru

    def add_encoder(self):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            encoder_inputs = self.add_embedding(self.inputs_placeholder)
            input_length = self.input_length_placeholder
            # Encoder
            encoder_cell = self.add_stacked_gru_cell()
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs,
                sequence_length=input_length,
                dtype=tf.float32)
            return encoder_outputs, encoder_state

    def add_decoder(self, encoder_outputs, encoder_state):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            decoder_inputs = self.add_embedding(self.decode_inputs_placeholder)
            decode_length = self.decode_length_placeholder
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_inputs, decode_length)
            # Decoder
            decoder_cell = self.add_stacked_gru_cell()
            output_layer = tf.layers.Dense(self.n_classes, use_bias=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, encoder_state,
                output_layer=output_layer)
            # Dynamic decoding
            (final_outputs, final_state,
             final_seq_length) = tf.contrib.seq2seq.dynamic_decode(decoder)
            return final_outputs, final_state, final_seq_length

    def add_test_decoder(self, encoder_outputs, encoder_state):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            # Helper
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.pretrained_embeddings,
                # 1 will lead a error
                tf.fill([2], self.word2index[self.utils.SOS]),
                self.word2index[self.utils.EOS])
            # Decoder
            decoder_cell = self.add_stacked_gru_cell()
            output_layer = tf.layers.Dense(self.n_classes, use_bias=False)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, helper, encoder_state,
                output_layer=output_layer)
            # Dynamic decoding
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=self.max_length)
            return outputs

    def add_test_prediction_op(self):
        encoder_outputs, encoder_state = self.add_encoder()
        decoder_outputs = self.add_test_decoder(
            encoder_outputs, encoder_state)
        return decoder_outputs.sample_id

    def add_train_prediction_op(self):
        encoder_outputs, encoder_state = self.add_encoder()
        decoder_outputs, _, _ = self.add_decoder(
            encoder_outputs, encoder_state)
        logits = decoder_outputs.rnn_output
        return logits

    def add_loss_op(self, logits):
        decoder_outputs = self.decode_outputs_placeholder
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)
        train_loss = tf.reduce_mean(crossent)
        # add L2 Regularzation
        trainable_variables = tf.trainable_variables(scope='encoder')
        reg_loss = self.l2_reg * \
            tf.reduce_sum([tf.nn.l2_loss(v)
                           for v in trainable_variables])
        trainable_variables = tf.trainable_variables(scope='decoder')
        reg_loss += self.l2_reg * \
            tf.reduce_sum([tf.nn.l2_loss(v)
                           for v in trainable_variables])
        return train_loss + reg_loss

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        # grad clip
        gradients, self.grad_norm = tf.clip_by_global_norm(
            gradients, self.max_grad_norm)
        update_step = optimizer.apply_gradients(
            zip(gradients, variables))
        return update_step

    def training_on_batch(self, sess, inputs_batch, input_length,
                          decode_length, decode_inputs, decode_outputs, dropout_rate):
        feed = self.create_feed_dict(
            inputs_batch, input_length, decode_length,
            decode_inputs, decode_outputs, dropout_rate)
        _, loss, grad_norm = sess.run(
            [self.train_op, self.loss, self.grad_norm], feed_dict=feed)
        return loss, grad_norm

    def generate_train_batch(self):
        train_size = self.inputs.shape[0]
        random_mask = np.random.choice(
            np.arange(0, train_size), self.batch_size)
        inputs_batch = self.inputs[random_mask]
        input_length = self.inputs_length[random_mask]
        decode_length = self.outputs_length[random_mask]
        decode_inputs_batch = self.decode_inputs[random_mask]
        decode_outputs_batch = self.decode_outputs[random_mask]
        return (inputs_batch, input_length, decode_length,
                decode_inputs_batch, decode_outputs_batch)

    def print_sample_id(self, sample_id):
        for s in sample_id:
            g = [self.word_list[ss] for ss in s]
            print(g[:10])

    def print_global_test(self, batch_size=5):
        data = self.utils.raw_data
        rid = random.sample(list(range(len(data))), batch_size)
        for idx in rid:
            print('Training Data:')
            print(data[idx])
            print('Model output:')
            inputs_batch = self.inputs[[idx]]
            inputs_length = self.inputs_length[[idx]]
            self.predict(inputs_batch, inputs_length)

    def predict(self, inputs_batch, inputs_length):
        batch_size = inputs_batch.shape[0]
        sample_ids = []
        for idx in range(batch_size):
            cur_inputs = [inputs_batch[idx]] * 2
            cur_inputs_length = [inputs_length[idx]] * 2
            feed_dict = self.create_feed_dict(
                cur_inputs, cur_inputs_length, None,
                None, None, dropout_rate=1)
            sample_id = self.prediction
            sample_id = self.sess.run(sample_id, feed_dict=feed_dict)
            sample_ids.append(sample_id[0])
        self.print_sample_id(sample_ids)
        return sample_ids

    def save(self):
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, self.save_path)
        print("Model saved in path: %s" % save_path)

    def play(self, left_roll):
        _, inputs_seq = self.utils.convert_cn_sentence(
            left_roll, reverse=True)
        length = len(inputs_seq)
        inputs = np.zeros((1, self.max_length), dtype=np.int32)
        inputs[0, 0:len(inputs_seq)] = inputs_seq
        self.predict(inputs, np.array([length]))
