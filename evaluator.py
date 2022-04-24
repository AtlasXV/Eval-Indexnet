import os
import time
import utils
import numpy
import cv2

class Eval_thread():
    def __init__(self, loader, method, dataset, output_dir, cuda):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.output_dir = output_dir
        self.logfile = os.path.join(output_dir, 'result.txt')

    def run(self):
        start_time = time.time()
        sad = self.Eval_sad()
        mse = self.Eval_mse()
        grad = self.Eval_grad()
        conn = self.Eval_conn()

        self.LOG(
            '{} ({}): {:.4f} sad || {:.4f} mse|| {:.4f} grad || {:.4f} conn\n'
            .format(self.dataset, self.method, sad, mse, grad, conn))
        return '[cost:{:.4f}s] {} ({}): {:.4f} sad || {:.4f} mse || {:.4f} grad || {:.4f} conn'.format(
            time.time() - start_time, self.dataset, self.method, sad, mse, grad, conn)

    def Eval_sad(self):
        print('eval[SAD]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_sad, img_num = 0.0, 0.0
        for pred, gt in self.loader:
            predImage = numpy.array(pred)
            gtImage = numpy.array(gt)
            sad = utils.compute_sad_loss(predImage, gtImage, gtImage)
            avg_sad += sad
            img_num += 1.0
        avg_sad /= img_num
        return avg_sad.item()

    def Eval_mse(self):
        print('eval[MSE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_mse, img_num = 0.0, 0.0
        for pred, gt in self.loader:
            predImage = numpy.array(pred)
            gtImage = numpy.array(gt)
            mse = utils.compute_mse_loss(predImage, gtImage, gtImage)
            avg_mse += mse
            img_num += 1.0
        avg_mse /= img_num
        return avg_mse.item()

    def Eval_grad(self):
        print('eval[Gradient]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_grad, img_num = 0.0, 0.0
        for pred, gt in self.loader:
            predImage = numpy.array(pred)
            gtImage = numpy.array(gt)
            grad = utils.compute_gradient_loss(predImage, gtImage, gtImage)
            avg_grad += grad / 1000
            img_num += 1.0
        avg_grad /= img_num
        return avg_grad.item()

    def Eval_conn(self):
        print('eval[Connectivity]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_conn, img_num = 0.0, 0.0
        for pred, gt in self.loader:
            predImage = numpy.array(pred)
            gtImage = numpy.array(gt)   
            conn = utils.compute_connectivity_loss(predImage, gtImage, gtImage, 0.1)
            avg_conn += conn / 1000
            img_num += 1.0
        avg_conn /= img_num
        return avg_conn.item()

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

