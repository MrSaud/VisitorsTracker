"""Shared-token authentication for machine clients.

Deliberately simple: one token in the environment, compared in constant
time. Operators in a browser authenticate with their normal Django staff
session instead, so nobody has to paste a token into a dashboard.
"""

import secrets

from django.conf import settings
from rest_framework import authentication, exceptions

HEADER_PREFIX = "Token"


class ServiceTokenUser:
    """A non-database principal representing a valid service token."""

    is_service_token = True
    is_staff = True
    username = "service-token"

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def __str__(self):
        return self.username


class StaticTokenAuthentication(authentication.BaseAuthentication):
    """Accepts ``Authorization: Token <COUNTER_API_TOKEN>``."""

    def authenticate(self, request):
        header = authentication.get_authorization_header(request).split()
        if not header or header[0].decode().lower() != HEADER_PREFIX.lower():
            return None
        if len(header) != 2:
            raise exceptions.AuthenticationFailed(
                "Authorization header must be 'Token <value>'.")

        configured = (getattr(settings, "COUNTER_API_TOKEN", "") or "").strip()
        if not configured:
            # Refuse rather than silently accept when no token is configured.
            raise exceptions.AuthenticationFailed(
                "Token auth is not configured on this server (COUNTER_API_TOKEN).")

        presented = header[1].decode()
        if not secrets.compare_digest(presented, configured):
            raise exceptions.AuthenticationFailed("Invalid token.")

        return (ServiceTokenUser(), None)

    def authenticate_header(self, request):
        return HEADER_PREFIX
