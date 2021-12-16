# global imports
from progress.bar import Bar

class AverageMeter(object):
    """
    Utility class to compute and store average and current values of a 
    variable. Class is copied from:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: ' + '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class _CustomIncrementalBar(Bar):
    phases = (' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█')

    def update(self):
        nphases = len(self.phases)
        filled_len = self.width * self.progress
        nfull = int(filled_len)                      # Number of full chars
        phase = int((filled_len - nfull) * nphases)  # Phase of last char
        nempty = self.width - nfull                  # Number of empty chars

        message = self.message % self
        bar = self.phases[-1] * nfull
        current = self.phases[phase] if phase > 0 else ''
        empty = self.empty_fill * max(0, nempty - len(current))
        suffix = self.suffix % self
        line = ''.join([message, self.bar_prefix, bar, current, empty,
                        self.bar_suffix, suffix])
        print(line, end='')


class ProgressMeter(_CustomIncrementalBar):
    """
    Utility class to display a progression bar with batch infos.
    """

    def __init__(self, num_batches, freq, meters, prefix=""):
        super().__init__(prefix, max=int(num_batches//freq)+1)
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters

    def display(self, batch):
        # build message to print
        entries = [self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += ['Total: {}'.format(self.elapsed_td)]
        entries += ['ETA: {}'.format(self.eta_td)]
        self.suffix = ' | '.join(entries) + '\n'
        self.next()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '(' + fmt + '/' + fmt.format(num_batches) + ')'