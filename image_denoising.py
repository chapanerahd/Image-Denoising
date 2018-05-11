import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os

def logistic(x):
    """
    Takes a data point
    and returns its logistic function value
    """
    return 1 / (1 + (1 / np.exp(x)))


def d_logistic(x):
    """
    Takes a data point
    and returns the differentiated logistic function value
    """
    return logistic(x) * (1 - logistic(x))


def process_image(image, f_size, f_values,  p_size, p_values):
    """
    Takes image, filter size, filter values, patch size and patch values as arguments
    and returns vectorized patch concatenating the column vectors horizontally
    where patches are collected from the top-left corner to the
    bottom-right corner of your input image by shifting it by one pixel each
    time
    """
    patched_image = np.zeros((f_values, p_values))
    val = 0
    for i in range(0, p_size):
        for j in range(0, p_size):
            patch = np.reshape(np.array(image[i : i + f_size, j : j + f_size]), \
                               (f_values,1))
            patched_image[:, val:(val+1)] = patch
            val = val + 1
    return patched_image


def set_parameters(f_size, image_shape):
    """
    Takes filter_size and image shape
    and returns total filter value, patch size and total patch values
    """
    filter_value = f_size * f_size
    patch_size = image_shape - filter_size + 1
    patch_value = patch_size * patch_size
    return filter_value, patch_size, patch_value


def init_filter(f_values, f_type):
    """
    Takes filter size and filter type
    and initializes the filter according to the tpye
    """
    if f_type == "zero":
        f = np.zeros((f_values, 1))
    else:
        f = np.random.rand(f_values, 1)
    return f


def gradient_descent(f, sgx_train_shape, sg_train, p_size, ro, iter):
    """
    Takes initialized filter, clean image, nosiy version of clean image
    patch size, learning rate as parameters
    and using gradient descent return a new filter and errors
    """
    p_value = p_size * p_size
    sg_train_ = sg_train[0:p_size, 0:p_size]
    sg_train_shape = np.reshape(sg_train_, (1, p_value))
    gd_errors = np.zeros(iter)
    for i in range(iter):
        ftx = np.dot(np.transpose(f), sgx_train_shape)
        err = sg_train_shape - logistic(ftx)
        gd_errors[i] = ((np.dot(err, np.transpose(err))) / p_value)
        val = err * d_logistic(ftx)
        dedf = - 2 * np.dot(sgx_train_shape, val.transpose()) / p_value
        nabala_f = -dedf
        f = f + ro * nabala_f
    return f, gd_errors


def plot_image(image, title):
    """
    Takes image and title
    and displays them
    """
    plt.grid(False)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def denoise_image(n_filter, p_size, noisy_image):
    """
    Takes filter, patch size and noisy image as arguments
    and return a image after applying that filter
    """
    denoise_img = np.dot(np.transpose(n_filter), noisy_image)
    reshaped_denoise_image = np.reshape(denoise_img, (p_size, p_size))
    return reshaped_denoise_image


if __name__ == "__main__":
    # Read the normalized data

    sgx_train = scipy.misc.imread("data/sgx_train.jpg") / 255
    sg_train = scipy.misc.imread("data/sg_train.jpg") / 255
    sgx_test = scipy.misc.imread("data/sgx_test.jpg") / 255

    # Plot the data

    plot_image(sgx_train, "Input training noisy Image")
    plot_image(sg_train, "Clean Image")
    plot_image(sgx_test, "Test Image")

    # Set filter size, learning rate  and number of iterations
    filter_size = 3
    learning_rate = 0.01
    iterations = 10000
    filter_type = "random"

    # Get other parameters according to filter size
    filter_values, patch_size, patch_values = set_parameters(filter_size, sgx_train.shape[0])

    # Initialize the filter
    f = init_filter(filter_values, filter_type)

    # Prepare the training and test data
    sgx_train_patched = process_image(sgx_train, filter_size, filter_values, patch_size, patch_values)
    sgx_test_patched = process_image(sgx_test, filter_size, filter_values, patch_size, patch_values)
    # f = np.zeros((9,1))

    # Applying gradient descent
    new_filter, errors = gradient_descent(f, sgx_train_patched, sg_train, patch_size, learning_rate, iterations)

    # Get denoise image
    denoise_sgx_train = denoise_image(new_filter, patch_size, sgx_train_patched)
    denoise_sgx_test = denoise_image(new_filter, patch_size, sgx_test_patched)
    if not os.path.exists("gradient_descent_output"):
        os.makedirs("gradient_descent_output")
    scipy.misc.imsave('gradient_descent_output/denoise_train.jpg', denoise_sgx_train)
    scipy.misc.imsave('gradient_descent_output/denoise_test.jpg', denoise_sgx_test)

    # Disply the denoise image
    plot_image(denoise_sgx_train, "Denoise training Image")
    plot_image(denoise_sgx_test, "Denoise test Image")
    #plt.im
    # Display errors
    plt.plot(errors)
    plt.show()

