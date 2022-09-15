import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from .animator3d import MedicalImageAnimator
from transfer_to_npy import load_mhd
import SimpleITK as sitk


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0:  # show images in the browser
            # if False:
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    # image_numpy = np.flipud(image_numpy)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if len(image_numpy.shape) == 4:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.gif' % (epoch, label))
                    animator = MedicalImageAnimator(image_numpy[0], [], 0, save=True)
                    animate = animator.run(img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)
            # update website
            # webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            # for n in range(epoch, 0, -1):
            #     webpage.add_header('epoch [%d]' % n)
            #     ims = []
            #     txts = []
            #     links = []
            #
            #     for label, image_numpy in visuals.items():
            #         img_path = 'epoch%.3d_%s.gif' % (n, label)
            #         ims.append(img_path)
            #         txts.append(label)
            #         links.append(img_path)
            #     webpage.add_images(ims, txts, links, width=self.win_size)
            # webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def toNii(self, save_dir, file_name, img_array, pixel_spacing, origin, direction):
        savedImg = sitk.GetImageFromArray(img_array)
        savedImg.SetOrigin(origin)
        savedImg.SetDirection(direction)
        savedImg.SetSpacing(pixel_spacing)
        # name = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + file_name
        sitk.WriteImage(savedImg, os.path.join(save_dir, file_name))

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, filename):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        _, spacing, orientation, origin, _, direction = load_mhd(
            '/mnt/data/czy/haimiandou/data/testB_mhd/' + filename)
        # _, spacing, orientation, origin, _, direction = load_mhd(
        #     '/mnt/data/czy/haimiandou/data/cvA_ori/' + filename)
        # print(visuals.items())
        image_numpy = visuals['fake_B'][0]
        image_numpy = image_numpy * 1000 - 200

        self.toNii("/mnt/data/czy/haimiandou/data/test-output/", filename, image_numpy, spacing, origin,
                   direction)
        # print(image_numpy)
        # print(image_numpy.shape)
        # for label, image_numpy in visuals.items():
        #     print(label)
        #     print(image_numpy.shape)
        #     # image_numpy = image_numpy[:, :, :, :64][0]
        #     # print(image_numpy.shape)
        #     image_name = '%s_%s.mhd' % (name, label)
        #     save_path = os.path.join(image_dir, image_name)
        #
        #     # util.save_image(image_numpy, save_path)
        #
        #     ims.append(image_name)
        #     txts.append(label)
        #     links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)
