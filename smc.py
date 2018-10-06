#!/usr/bin/env python3

import combinators
import importance

class SequentialMonteCarlo(combinators.Model):
    def __init__(self, step_model, T, step_proposal=None,
                 initializer=None, resample_factor=2):
        resampled_step = importance.ImportanceResampler(
            step_model, step_proposal, resample_factor=resample_factor
        )
        step_sequence = combinators.Model.sequence(resampled_step, T)
        if initializer:
            model = combinators.Model.compose(step_sequence, initializer,
                                              intermediate_name='initializer')
        else:
            model = step_sequence
        super(SequentialMonteCarlo, self).__init__(model)
        self.resampled_step = resampled_step
        self.initializer = initializer
