import os
import time
import utils

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
            sad = utils.compute_sad_loss(pred, gt, gt)
            avg_sad += sad
            img_num += 1.0
        avg_sad /= img_num
        return avg_sad.item()

    def Eval_mse(self):
        print('eval[MSE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_sad, img_num = 0.0, 0.0
        for pred, gt in self.loader:
            sad = utils.compute_mse_loss(pred, gt, gt)
            avg_sad += sad
            img_num += 1.0
        avg_sad /= img_num
        return avg_sad.item()

    def Eval_grad(self):
        print('eval[MSE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_sad, img_num = 0.0, 0.0
        for pred, gt in self.loader:
            sad = utils.compute_gradient_loss(pred, gt, gt)
            avg_sad += sad
            img_num += 1.0
        avg_sad /= img_num
        return avg_sad.item()

    def Eval_conn(self):
        print('eval[MSE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_sad, img_num = 0.0, 0.0
        for pred, gt in self.loader:
            sad = utils.compute_connectivity_loss(pred, gt, gt)
            avg_sad += sad
            img_num += 1.0
        avg_sad /= img_num
        return avg_sad.item()

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

