"""Opt-in natural-campaign observation timing matched to the training environment."""
from . import env

_Runtime = env.Runtime
_Observer = env.Observer
_active = None


class Runtime(_Runtime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global _active
        _active = self


class Observer(_Observer):
    def reset(self, screen):
        self.pending = None
        super().reset(screen)

    def push(self, screen, action):
        # Campaign pushes before MatchTracker.settle(). Training pushes after it.
        # Commit exactly one observation at the next decision, using the live screen.
        self.pending = action

    def obs(self):
        if self.pending is not None:
            super().push(_active.screen, self.pending)
            self.pending = None
        return super().obs()


def install():
    env.Runtime, env.Observer = Runtime, Observer


def uninstall():
    env.Runtime, env.Observer = _Runtime, _Observer
