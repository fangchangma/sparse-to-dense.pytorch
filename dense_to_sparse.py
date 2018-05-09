import numpy as np
import cv2


def rgb2grayscale(rgb):
    return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, rgb, depth):
        pass

    def __repr__(self):
        pass


class DSOSampling(DenseToSparse):
    name = "dso"

    def __init__(self, num_samples, grad_th, window_size, sub_window_size):
        self.num_samples = num_samples
        self.grad_th = grad_th
        self.window_size = window_size
        self.sub_window_size = sub_window_size

    def __repr__(self):
        return "%s{ns=%d,grad_th=%f,window_size=%f,sub_window_size=%f}" % (self.name, self.num_samples, self.grad_th, self.window_size, self.sub_window_size)

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(h // nrows, w // ncols, nrows, ncols))

    def invblockshaped(self, arr, h, w):
        return (arr.swapaxes(1, 2)
                .reshape(h, w))

    def blockblockshaped(self, arr, nrows, ncols):
        h, w, r1, c1 = arr.shape
        return (arr.reshape(h, w, r1 // nrows, nrows, -1, ncols)
                .swapaxes(3, 4)
                .reshape(h, w, r1 // nrows, c1 // ncols, nrows, ncols))

    def invblockblockshaped(self, arr, h, w, r1, c1):
        return (arr.swapaxes(3, 4)
                .reshape(h, w, r1, c1))

    def dense_to_sparse(self, rgb, depth):
        gray = rgb2grayscale(rgb)

        height = gray.shape[0]
        width = gray.shape[1]

        laplacian = abs(cv2.Laplacian(gray, cv2.CV_64F))
        mask = np.full((height, width), False).astype("bool")

        submatrices = self.blockshaped(laplacian, self.window_size,
                                       self.window_size)
        gradAbs = submatrices.reshape(height * width // self.window_size // self.window_size, self.window_size * self.window_size)
        gradAbs = np.median(gradAbs, axis=1)
        gradAbs = gradAbs.reshape(height // self.window_size, width // self.window_size)

        sub_window_size = self.sub_window_size
        grad_th = self.grad_th
        for x in range(0, 3):

            pixel_mask = np.full((height // self.window_size, width // self.window_size, self.window_size, self.window_size), False).astype(
                "bool")

            maxima = self.blockblockshaped(submatrices, sub_window_size, sub_window_size)
            i, j, k, l, m, n = maxima.shape
            maxima = maxima.reshape(i, j, k, l, m * n)
            max_mask = maxima.max(axis=4, keepdims=1) == maxima
            max_mask = max_mask.reshape(i, j, k, l, m, n)
            max_mask = self.invblockblockshaped(max_mask, height // self.window_size, width // self.window_size, self.window_size,
                                                self.window_size)
            max_mask = self.invblockshaped(max_mask, height, width)

            for i in range(0, height // self.window_size):
                for j in range(0, width // self.window_size):
                    pixel_mask[i, j] = submatrices[i, j] >= (gradAbs[i, j] + grad_th)

            pixel_mask = self.invblockshaped(pixel_mask, height, width)

            mask = mask | (pixel_mask & max_mask)

            sub_window_size = sub_window_size * 2
            grad_th = grad_th - 1

        prob = float(self.num_samples) / np.count_nonzero(mask)

        mask_keep = np.random.uniform(0, 1, mask.shape) < prob
        sparse_depth_mask = np.full(mask.shape, False).astype("bool")
        sparse_depth_mask[mask_keep] = mask[mask_keep]

        return sparse_depth_mask

class UniformSampling(DenseToSparse):
    name = "uar"

    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, rgb, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        if self.max_depth is not np.inf:
            mask_keep = depth <= self.max_depth
            n_keep = np.count_nonzero(mask_keep)
            if n_keep == 0:
                return mask_keep
            else:
                prob = float(self.num_samples) / n_keep
                return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)
        else:
            prob = float(self.num_samples) / depth.size
            return np.random.uniform(0, 1, depth.shape) < prob


class SimulatedStereo(DenseToSparse):
    name = "sim_stereo"

    def __init__(self, num_samples, max_depth=np.inf, dilate_kernel=3, dilate_iterations=1):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.dilate_kernel = dilate_kernel
        self.dilate_iterations = dilate_iterations

    def __repr__(self):
        return "%s{ns=%d,md=%f,dil=%d.%d}" % \
               (self.name, self.num_samples, self.max_depth, self.dilate_kernel, self.dilate_iterations)

    # We do not use cv2.Canny, since that applies non max suppression
    # So we simply do
    # RGB to intensitities
    # Smooth with gaussian
    # Take simple sobel gradients
    # Threshold the edge gradient
    # Dilatate
    def dense_to_sparse(self, rgb, depth):
        gray = rgb2grayscale(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        depth_mask = np.bitwise_and(depth != 0.0, depth <= self.max_depth)

        edge_fraction = float(self.num_samples) / np.size(depth)

        mag = cv2.magnitude(gx, gy)
        min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
        mag_mask = mag >= min_mag

        if self.dilate_iterations >= 0:
            kernel = np.ones((self.dilate_kernel, self.dilate_kernel), dtype=np.uint8)
            cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=self.dilate_iterations)

        mask = np.bitwise_and(mag_mask, depth_mask)
        return mask
