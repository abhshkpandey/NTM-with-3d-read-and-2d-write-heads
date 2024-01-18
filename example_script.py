from ntm_model import NTM, SharedReadWriteHeads
import tensorflow as tf

def user_interaction(user_input_sequence, ntm_model):
    # Interact with the NTM model using user input
    input_sequence = tf.constant([user_input_sequence], dtype=tf.float32)

    # Run the NTM on the input sequence
    ntm_output_sequence, ntm_final_state = ntm_model(input_sequence)

    # Extract relevant information for the user
    controller_output = ntm_final_state[0]
    read_vectors = ntm_final_state[1:]

    return controller_output.numpy(), read_vectors

def main():
    # Parameters
    memory_size = 128
    memory_vector_dim = 20
    controller_units = 100
    num_read_heads = 4
    num_write_heads = 4

    # Create NTM instance
    ntm = NTM(memory_size, memory_vector_dim, controller_units, num_read_heads, num_write_heads)

    # User input sequence
    user_input_sequence = [1.0, 0.5, 0.2, 0.8]

    # Interact with the NTM
    controller_output, read_vectors = user_interaction(user_input_sequence, ntm)

    # Print results for the user
    print("Controller Output:", controller_output)
    print("Read Vectors:", read_vectors)

if __name__ == "__main__":
    main()
