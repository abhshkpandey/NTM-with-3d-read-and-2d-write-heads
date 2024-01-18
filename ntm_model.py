import tensorflow as tf

class SharedReadWriteHeads(tf.keras.layers.Layer):
    def __init__(self, memory_vector_dim, num_heads):
        super(SharedReadWriteHeads, self).__init__()

        self.num_heads = num_heads
        self.memory_vector_dim = memory_vector_dim

        # Shared read and write heads
        self.read_heads = [self.create_head() for _ in range(num_heads)]
        self.write_heads = [self.create_head() for _ in range(num_heads)]

    def create_head(self):
        return tf.keras.layers.Dense(self.memory_vector_dim + 1, activation='sigmoid')

class NTM(tf.keras.Model):
    def __init__(self, memory_size, memory_vector_dim, controller_units, num_read_heads, num_write_heads):
        super(NTM, self).__init__()

        # Parameters
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.controller_units = controller_units
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads

        # Shared Read and Write Heads
        self.shared_heads = SharedReadWriteHeads(memory_vector_dim, num_read_heads + num_write_heads)

        # Controller (2D LSTM)
        self.controller = tf.keras.layers.LSTMCell(self.controller_units)

        # External Memory
        self.memory = tf.Variable(tf.random.normal([self.memory_size, self.memory_vector_dim]))

    def address_memory(self, keys, strengths):
        content_weights = tf.nn.softmax(tf.matmul(self.memory, tf.transpose(keys)) * strengths, axis=1)
        return content_weights

    def read_memory(self, read_weights):
        return tf.reduce_sum(tf.expand_dims(read_weights, axis=-1) * self.memory, axis=1)

    def write_memory(self, write_weights, erase_vector, add_vector):
        self.memory = tf.tensor_scatter_nd_sub(self.memory, indices=tf.range(self.memory_size)[:, tf.newaxis],
                                              updates=(tf.transpose(write_weights)[:, :, tf.newaxis] * erase_vector))
        self.memory = tf.tensor_scatter_nd_add(self.memory, indices=tf.range(self.memory_size)[:, tf.newaxis],
                                              updates=(tf.transpose(write_weights)[:, :, tf.newaxis] * add_vector))

    def call(self, inputs, prev_state=None):
        if prev_state is None:
            prev_state = [tf.zeros([tf.shape(inputs)[0], self.controller_units]),
                          tf.zeros([tf.shape(inputs)[0], self.controller_units])]

        controller_input = tf.concat([inputs, prev_state[0]], axis=-1)
        controller_output, controller_state = self.controller(controller_input, prev_state)

        # Read Heads
        read_keys = [head(controller_output) for head in self.shared_heads.read_heads[:self.num_read_heads]]
        read_strengths = tf.nn.softmax(tf.stack([head(controller_output) for head in self.shared_heads.read_heads[:self.num_read_heads]], axis=-1), axis=-1)
        read_weights = self.address_memory(read_keys, read_strengths)
        read_vectors = [self.read_memory(weights) for weights in tf.unstack(read_weights, axis=-1)]

        # Write Heads
        write_keys = [head(controller_output) for head in self.shared_heads.write_heads[:self.num_write_heads]]
        write_strengths = tf.nn.softmax(tf.stack([head(controller_output) for head in self.shared_heads.write_heads[:self.num_write_heads]], axis=-1), axis=-1)
        write_weights = self.address_memory(write_keys, write_strengths)

        # Write to memory
        erase_vector = tf.nn.sigmoid(self.shared_heads.write_heads[self.num_write_heads](controller_output))
        add_vector = tf.nn.tanh(self.shared_heads.write_heads[self.num_write_heads + 1](controller_output))
        self.write_memory(write_weights, erase_vector, add_vector)

        return tf.concat([controller_output] + read_vectors, axis=-1), [controller_state] + read_vectors
