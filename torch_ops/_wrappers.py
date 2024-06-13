class register_bwd_op:
    def __init__(self, bwd_fn):
        self.bwd_fn = bwd_fn
    
    def __call__(self,fn):
        fn.bwd = self.bwd_fn
        return fn