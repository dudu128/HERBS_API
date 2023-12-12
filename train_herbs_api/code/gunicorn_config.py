import logging
deamon = False
bind = '0.0.0.0:8080'                           # port
pidfile = 'gunicorn.pid.2'                      # record gunicorn's pid
chdir = '.'                                     # main root
worker_class = 'uvicorn.workers.UvicornWorker'  # worker type
workers = 4                                     # process num
timeout = 600                            
preload_app = True                              # avoid always use the same process
loglevel = 'info'                               # loglevel : info/debug
accesslog = "gunicorn_access.log"               # log type
errorlog = "gunicorn_error.log"                 # log type
capture_output = True                           # store the output in the log

