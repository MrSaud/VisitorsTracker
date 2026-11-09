"""Gunicorn settings for VisitorsTracker.

The important line is `workers = 1`, and it is not about load.

Camera workers are threads inside the web process. Each gunicorn worker is
a separate process with its own registry, so with COUNTER_AUTOSTART every
one of them would start its own counter for the same camera. An RTSP stream
serves them all happily, so nothing errors -- the counts simply multiply.
Measured, before cameras took a lease: four processes over a clip with 3
crossings recorded 12.

The lease now refuses the duplicates, so extra workers would not corrupt the
counts. They still would not work: only the process holding the lease has a
live camera, so the live view, the start/stop buttons and the health tiles
would answer differently depending on which worker got the request.

One process, many threads, is the right shape for this app.
"""

import multiprocessing  # noqa: F401  (kept so the next person sees it is deliberate)

bind = "127.0.0.1:8000"

# See above. Do not raise this.
workers = 1

# Threads, not processes. An MJPEG live view holds its thread for as long as
# somebody watches, so allow for a few viewers plus ordinary requests.
worker_class = "gthread"
threads = 16

# A live view is a response that never ends; the default 30s would cut it.
timeout = 0
graceful_timeout = 30
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = "info"
# Keep the health endpoint out of the access log; it is polled constantly.
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(M)sms'

proc_name = "visitorstracker"
