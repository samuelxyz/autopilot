import numpy as np

class History(dict):
    '''Dictionary tracking the history of specified arbitrary floats or numpy arrays. 
    self[name] returns a numpy array with num_steps added as the first dimension.'''
    def __init__(self, num_steps, values_to_track: dict):
        super().__init__()
        self.num_steps = num_steps
        self.refs = {}
        self.history = {}
        for name, info in values_to_track.items():
            if isinstance(info, tuple):
                ref, shape = info
            else:
                ref = info
                value = ref() if callable(ref) else ref
                try:
                    shape = np.shape(value) # works for non-numpy numbers too
                except:
                    print(f'Could not identify shape to use for {ref} when creating history tracker - specify it in values_to_track as "name": (item, shape)')
            self.refs[name] = (ref, callable(ref))
            self[name] = np.zeros((num_steps,) + shape)
    
    def save_timestep(self, i):
        '''Access each of the tracked items, and write their current values into timestep i.'''
        for name, tracker_array in self.items():
            thing_to_be_written, should_be_called = self.refs[name]
            try:
                if should_be_called:
                    tracker_array[i] = thing_to_be_written()
                else:
                    tracker_array[i] = thing_to_be_written
            except:
                tracker_array[i] = np.nan
    