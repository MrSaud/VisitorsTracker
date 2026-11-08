from django.apps import AppConfig


class CounterConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "counter"
    verbose_name = "Visitor counting"

    def ready(self):
        # Opt-in: with COUNTER_AUTOSTART=1 the server starts every active
        # camera and keeps them running, so counting resumes after a reboot
        # or a restart without anyone pressing Start. Off by default, since
        # importing the app should not grab a camera.
        from .supervisor import maybe_autostart

        maybe_autostart()
