"""Independent high-level traffic-event generation; never executes vehicle control."""

from .inference import Director, DirectorResult

__all__ = ["Director", "DirectorResult"]
