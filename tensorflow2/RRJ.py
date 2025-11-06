# rRJ2.py
# Alternating algorithm for auto-encoder + low-rank Jacobian regularization (TensorFlow 2 version)
# Author: [Your Name]
# Date: [YYYY-MM-DD]

import tensorflow as tf
import numpy as np

@tf.function
def compute_static_flat_jacobian(model, x):
    x_shape = tf.shape(x)
    batch_size = x_shape[0]
    x_flat = tf.reshape(x, [batch_size, -1])

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_flat)
        y = model(tf.reshape(x_flat, tf.shape(x)))
        y_flat = tf.reshape(y, [batch_size, -1])

    J = tape.batch_jacobian(y_flat, x_flat)
    del tape
    return J

@tf.function
def compute_static_encoder_jacobian(model_encoder, x):
    x_shape = tf.shape(x)
    batch_size = x_shape[0]
    x_flat = tf.reshape(x, [batch_size, -1])

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_flat)
        y = model_encoder(tf.reshape(x_flat, tf.shape(x)))
        y_flat = tf.reshape(y, [batch_size, -1])

    J = tape.batch_jacobian(y_flat, x_flat)
    del tape
    return J

@tf.function
def compute_static_decoder_jacobian(model_decoder, x):
    x_shape = tf.shape(x)
    batch_size = x_shape[0]
    x_flat = tf.reshape(x, [batch_size, -1])

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_flat)
        y = model_decoder(tf.reshape(x_flat, tf.shape(x)))
        y_flat = tf.reshape(y, [batch_size, -1])

    J = tape.batch_jacobian(y_flat, x_flat)
    del tape
    return J

@tf.function
def train_step(model, x_batch, grad_batch, B_batch, optimizer, gamma, Lambda, epsilon):

    with tf.GradientTape() as tape:
        recon = model(x_batch)
        loss_rec = tf.reduce_mean(tf.square(x_batch - recon))

        # curvature / smoothness term
        noise = tf.random.normal(tf.shape(x_batch), stddev=epsilon)
        J1 = compute_static_flat_jacobian(model, x_batch)
        J2 = compute_static_flat_jacobian(model, x_batch + noise)
        grad_diff = J2 - J1
        loss_curv = gamma * tf.reduce_mean(tf.square(grad_diff))


        # rank penalty term (using Jacobian on grad_batch)
        J = compute_static_flat_jacobian(model, grad_batch)
        loss_pen = Lambda * tf.reduce_mean(tf.square(J - B_batch)) 

        loss = loss_rec + loss_curv + loss_pen

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function(reduce_retracing=True)
def tf_svd_product(A, U, S, VH):
    """
    SVD of A @ (U @ diag(S) @ VH) via QR+SVD.
    Returns (U_full, s, Vh_full) so that:
       A @ U @ diag(S) @ VH = U_full @ diag(s) @ Vh_full
    """
    S  = tf.linalg.diag(S)

    # QR of A @ U
    Q, R = tf.linalg.qr(tf.matmul(A, U), full_matrices=False)      # (n,d) , (d,d)

    # SVD of reduced core R @ diag(S)
    RS = tf.matmul(R, S)
    s, u_r, v_r = tf.linalg.svd(RS, full_matrices=False)           # RS = u_r @ diag(s) @ v_r^T

    U_full  = tf.matmul(Q, u_r)                   # (n,d)
    Vh_full = tf.matmul(tf.transpose(v_r), VH)    # (d, n) because v_r^T @ VH

    return U_full, s, Vh_full

@tf.function(reduce_retracing=True)
def batch_svd_product(elems, k):
    Jd, Ue, se, VeT = elems
    U_g, s_g, V_gT = tf_svd_product(Jd, Ue, se, VeT)
    U_k  = U_g[:, :k]
    S_k  = tf.linalg.diag(s_g[:k])
    V_kT = V_gT[:k, :]
    return tf.matmul(U_k, tf.matmul(S_k, V_kT))

@tf.function(reduce_retracing=True)
def batched_svd(J_d_batch, U_e, s_e, V_eT, k):
    return tf.map_fn(
        lambda elems: batch_svd_product(elems, k),
        (J_d_batch, U_e, s_e, V_eT),
        fn_output_signature=tf.float32,
        parallel_iterations=8,
    )

@tf.function(reduce_retracing=True)
def compute_truncated_svd_autoencoder(model_encoder, model_decoder, x_batch, k):
    """
    Batched version:
    Compute rank-k SVD of J_g(x) = J_d(e(x)) J_e(x)
    for each sample in the batch, using batched tf.linalg.svd.
    Returns a tensor [batch_size, n_out, n_in].
    """

    # Encode inputs
    z_batch = model_encoder(x_batch)

    # Compute batched Jacobians
    J_e_batch = compute_static_encoder_jacobian(model_encoder, x_batch)   # [B, d, n]
    J_d_batch = compute_static_decoder_jacobian(model_decoder, z_batch)   # [B, n, d]

    # Batched SVD of encoder Jacobians
    s_e, U_e, V_e = tf.linalg.svd(J_e_batch, full_matrices=False)  # [B, d], [B, d, d], [B, n, d]
    V_eT = tf.linalg.matrix_transpose(V_e)                         # [B, d, n]

    # Map across the batch in graph mode
    B_batch = batched_svd(J_d_batch, U_e, s_e, V_eT, k)
    return B_batch

@tf.function(reduce_retracing=True)
def recompute_B(model, dataset_grad, num_grad_batches, k):
    """
    Recompute B by iterating `num_grad_batches` batches from dataset_grad.

    Args:
      model: tf.keras.Model with .encoder and .decoder
      dataset_grad: a batched, drop_remainder=True dataset
      num_grad_batches: tf.int32 scalar tensor
      k: tf.int32 scalar tensor
    Returns:
      Tensor of shape [num_grad_batches*batch_size, n_out, n_in]
    """
    # Create iterator inside graph so its elements are in-scope.
    ds_iter = iter(dataset_grad)

    # Accumulate per-batch results with a TensorArray (graph-safe).
    ta = tf.TensorArray(tf.float32, size=num_grad_batches)

    # Loop entirely in graph mode.
    for i in tf.range(num_grad_batches):
        grad_batch = next(ds_iter)  # safe in autograph
        B_batch_new = compute_truncated_svd_autoencoder(
            model.encoder, model.decoder, grad_batch, k)
        ta = ta.write(i, B_batch_new)

    # Concatenate along batch dimension: [num_grad_batches, batch, n_out, n_in] -> [num_grad_batches*batch, n_out, n_in]
    B_all = ta.stack()
    B_all = tf.reshape(B_all, [-1, tf.shape(B_all)[-2], tf.shape(B_all)[-1]])
    return B_all


def alternating_train(model, x_train, grad_x_train,
                      k=30, batch_size=32, T=10, steps_per_iter=100,
                      gamma=1.0, Lambda=10.0, epsilon=0.1, learning_rate=1e-4):
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Build datasets for the two different sampling streams
    dataset_main = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size, drop_remainder=True)
    dataset_grad = tf.data.Dataset.from_tensor_slices(grad_x_train).shuffle(10000).batch(batch_size, drop_remainder=True).repeat()

    # Initialize B-matrices (low-rank Jacobian approximations)
    sample_batch = next(iter(dataset_grad.take(1)))
    n_in = int(np.prod(sample_batch.shape[1:]))
    n_out = int(np.prod(model(sample_batch[:1]).shape[1:]))
    num_grad_batches = int(np.floor(grad_x_train.shape[0] / batch_size))
    B = tf.zeros([num_grad_batches*batch_size, n_out, n_in], dtype=tf.float32)


    # Main alternating loop
    for t in range(T):
        print(f"=== Outer iteration {t+1}/{T} ===")
        step = 0

        # θ-update loop (optimize neural parameters given current B)
        for (x_batch, grad_batch) in zip(
                dataset_main.take(steps_per_iter),
                dataset_grad):

            # Slice the corresponding B_batch for this grad_batch
            start_idx = (step * batch_size) % B.shape[0]
            end_idx   = start_idx + batch_size
            if end_idx > B.shape[0]: #it will never happen, due to drop_remainder=True, but I maintain that line
                part1 = B[start_idx:]
                part2 = B[:end_idx - B.shape[0]]
                B_batch = tf.concat([part1, part2], axis=0)
            else:
                B_batch = B[start_idx:end_idx]

            loss = train_step(model, x_batch, grad_batch, B_batch,
                              optimizer, gamma, Lambda, epsilon)

            if step % 20 == 0:
                tf.print(" step", step, "loss", loss)
            step += 1

        # === B-update: recompute truncated SVD of new Jacobians ===
        print("Recomputing B (rank-k truncated SVDs)...")
        B = recompute_B(model, dataset_grad, num_grad_batches, k)

    print("Training complete.")
    return model, B
