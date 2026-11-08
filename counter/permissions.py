"""Who may read, and who may change things.

Reads are open by default: a wall display or dashboard should not need a
credential. Anything that changes state -- starting or stopping a camera,
and especially resetting counts, which deletes history -- requires either a
Django staff session or the service token.
"""

from django.conf import settings
from rest_framework import permissions

SAFE_METHODS = permissions.SAFE_METHODS


def _is_operator(request) -> bool:
    user = getattr(request, "user", None)
    if user is None or not user.is_authenticated:
        return False
    return bool(getattr(user, "is_staff", False))


class IsOperatorOrReadOnly(permissions.BasePermission):
    """Open reads (configurable), authenticated writes."""

    message = (
        "This action changes server state. Authenticate as a staff user, or "
        "send 'Authorization: Token <COUNTER_API_TOKEN>'."
    )

    def has_permission(self, request, view):
        if request.method in SAFE_METHODS:
            if getattr(settings, "COUNTER_PUBLIC_READS", True):
                return True
            return _is_operator(request)
        return _is_operator(request)
