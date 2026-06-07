import sys
import itertools

class ProgressCallback:
    def __init__(self, current_step, total_steps, info_str, total_epochs):
        self.current_step = current_step
        self.total_steps = total_steps
        self.info_str = info_str
        self.total_epochs = total_epochs
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']) 

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 15 == 0 or epoch == self.total_epochs - 1:
            spin = next(self.spinner)
            sys.stdout.write(f"\r {spin} Entrenando [{self.current_step}/{self.total_steps}] -> {self.info_str} | Época: {epoch+1}/{self.total_epochs}   ")
            sys.stdout.flush()