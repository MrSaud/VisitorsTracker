#!/bin/bash
# Remove VisitorsTracker from a Linux server, and nothing else.
#
# This exists because the app was installed onto a machine that already had
# a job. Stopping or rebuilding that machine is not an option, so every path
# below is named explicitly and anything unrecognised is left alone.
#
#   sudo bash deploy/uninstall.sh          # say what would go, change nothing
#   sudo bash deploy/uninstall.sh --yes    # actually remove it
#
# The database and the environment file are copied into a tarball before
# anything is deleted; the path is printed at the end.
set -uo pipefail

APPLY=no
BACKUP_DIR=/root
for arg in "$@"; do
  case "$arg" in
    --yes|-y)      APPLY=yes ;;
    --backup-dir=*) BACKUP_DIR="${arg#*=}" ;;
    -h|--help)     sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

[ "$(id -u)" = "0" ] || { echo "Run this with sudo." >&2; exit 1; }

# Everything this app installs. Nothing outside this list is touched.
#
# The paths and the commands are overridable so this can be exercised against
# a fake tree in the test suite. A script that deletes things on a shared
# server should be tested, and it cannot be tested if it can only ever point
# at the real /opt and the real systemctl.
UNITS=(visitorstracker.service visitorstracker-backup.service visitorstracker-backup.timer)
ENV_FILE="${VT_ENV_FILE:-/etc/visitorstracker.env}"
APP_DIR="${VT_APP_DIR:-/opt/visitorstracker}"
UNIT_DIR="${VT_UNIT_DIR:-/etc/systemd/system}"
SVC_USER="${VT_SVC_USER:-visitorstracker}"
NGINX_DIRS=(${VT_NGINX_DIRS:-/etc/nginx/conf.d /etc/nginx/sites-available /etc/nginx/sites-enabled})
SYSTEMCTL="${VT_SYSTEMCTL:-systemctl}"
NGINX="${VT_NGINX:-nginx}"

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
act()  { if [ "$APPLY" = yes ]; then "$@"; else say "   would run: $*"; fi; }

if [ "$APPLY" != yes ]; then
  say "DRY RUN -- nothing will be changed. Add --yes to go ahead."
fi

step "what is on this machine now"
say " other services running (none of these will be touched):"
"$SYSTEMCTL" list-units --type=service --state=running --no-legend --no-pager 2>/dev/null \
  | awk '{print "   " $1}' | grep -v visitorstracker || say "   (could not list)"

step "VisitorsTracker pieces found"
FOUND=no
for unit in "${UNITS[@]}"; do
  if [ -f "$UNIT_DIR/$unit" ]; then say " unit        $UNIT_DIR/$unit"; FOUND=yes; fi
done
[ -f "$ENV_FILE" ] && { say " env         $ENV_FILE"; FOUND=yes; }
[ -d "$APP_DIR" ]  && { say " directory   $APP_DIR ($(du -sh "$APP_DIR" 2>/dev/null | cut -f1))"; FOUND=yes; }
id "$SVC_USER" >/dev/null 2>&1 && { say " user        $SVC_USER"; FOUND=yes; }

# Find the nginx fragment by what is in it, not by its filename: whoever
# installed it may have renamed it, and guessing a name risks deleting
# somebody else's site.
NGINX_FILES=()
for dir in "${NGINX_DIRS[@]}"; do
  [ -d "$dir" ] || continue
  while IFS= read -r f; do
    grep -lqs -e 'upstream visitorstracker' -e 'visitorstracker' "$f" 2>/dev/null && NGINX_FILES+=("$f")
  done < <(find "$dir" -maxdepth 1 -type f -o -maxdepth 1 -type l 2>/dev/null)
done
for f in "${NGINX_FILES[@]:-}"; do [ -n "$f" ] && { say " nginx       $f"; FOUND=yes; }; done

if [ "$FOUND" = no ]; then
  say " nothing found -- it is not installed here, or it lives somewhere else."
  exit 0
fi

step "saving the database and config first"
STAMP="$(date +%Y%m%d-%H%M%S)"
TARBALL="$BACKUP_DIR/visitorstracker-removed-$STAMP.tar.gz"
# Honour a relocated data directory if the env file names one.
DATA_DIR="$APP_DIR"
if [ -f "$ENV_FILE" ]; then
  FROM_ENV="$(grep -E '^COUNTER_DATA_DIR=' "$ENV_FILE" 2>/dev/null | cut -d= -f2-)"
  [ -n "${FROM_ENV:-}" ] && DATA_DIR="$FROM_ENV"
fi
SAVE=()
for candidate in "$DATA_DIR"/db.sqlite3 "$DATA_DIR"/backups "$DATA_DIR"/.secret_key "$ENV_FILE"; do
  [ -e "$candidate" ] && SAVE+=("$candidate")
done
SAVED=no
if [ ${#SAVE[@]} -eq 0 ]; then
  say " nothing to save -- no database or config found"
else
  for s in "${SAVE[@]}"; do say "   $s"; done
  if [ "$APPLY" = yes ]; then
    # `-C /` with paths relative to it, rather than GNU tar's
    # --absolute-names: BSD tar rejects that option outright, and the first
    # version of this script announced "saved to ..." anyway and went on to
    # delete the database. Hence the verification below.
    REL=()
    for s in "${SAVE[@]}"; do REL+=("${s#/}"); done
    tar -czf "$TARBALL" -C / "${REL[@]}" 2>"$TARBALL.err"

    if [ -s "$TARBALL" ] && tar -tzf "$TARBALL" >/dev/null 2>&1; then
      say " saved to $TARBALL ($(du -h "$TARBALL" 2>/dev/null | cut -f1))"
      SAVED=yes
      rm -f "$TARBALL.err"
    else
      say ""
      say " COULD NOT SAVE a copy to $TARBALL:"
      [ -s "$TARBALL.err" ] && sed 's/^/   /' "$TARBALL.err"
      say ""
      say " Nothing has been removed. The point of this step is that the"
      say " counting history survives the uninstall, so carrying on without"
      say " it would be worse than stopping here. Fix the backup path"
      say " (--backup-dir=...) and run it again."
      exit 1
    fi
  else
    say "   would save them to $TARBALL, and stop if that failed"
  fi
fi

step "stopping it"
# The timer first, so a backup cannot start while the service is going away.
for unit in visitorstracker-backup.timer visitorstracker-backup.service visitorstracker.service; do
  if "$SYSTEMCTL" list-unit-files "$unit" >/dev/null 2>&1 && [ -f "$UNIT_DIR/$unit" ]; then
    act "$SYSTEMCTL" disable --now "$unit"
  fi
done

step "removing the unit files"
for unit in "${UNITS[@]}"; do
  [ -f "$UNIT_DIR/$unit" ] && act rm -f "$UNIT_DIR/$unit"
done
act "$SYSTEMCTL" daemon-reload
act "$SYSTEMCTL" reset-failed

step "removing the nginx site"
# Move it aside, check nginx still accepts its configuration, and only then
# reload. If the check fails the file goes back and nginx is left alone --
# this machine serves other things through it.
for f in "${NGINX_FILES[@]:-}"; do
  [ -n "$f" ] || continue
  if [ "$APPLY" = yes ]; then
    mkdir -p "$BACKUP_DIR/nginx-$STAMP"
    mv "$f" "$BACKUP_DIR/nginx-$STAMP/$(basename "$f")"
    if "$NGINX" -t >/dev/null 2>&1; then
      "$SYSTEMCTL" reload nginx && say " removed $f and reloaded nginx"
    else
      mv "$BACKUP_DIR/nginx-$STAMP/$(basename "$f")" "$f"
      say " PUT BACK $f -- nginx -t failed without it, so something else"
      say " references it. nginx was not reloaded. Look into it by hand:"
      "$NGINX" -t 2>&1 | sed 's/^/   /'
    fi
  else
    say "   would move aside: $f, then nginx -t, then reload only if it passes"
  fi
done

step "removing the application"
[ -f "$ENV_FILE" ] && act rm -f "$ENV_FILE"
[ -d "$APP_DIR" ]  && act rm -rf "$APP_DIR"

step "removing the service account"
if id "$SVC_USER" >/dev/null 2>&1; then
  # Only if it owns nothing outside the directory just removed.
  STRAY="$(find / -xdev -user "$SVC_USER" -not -path "$APP_DIR/*" 2>/dev/null | head -5)"
  if [ -n "$STRAY" ]; then
    say " keeping $SVC_USER -- it still owns files elsewhere:"
    printf '   %s\n' $STRAY
  else
    act userdel "$SVC_USER"
    getent group "$SVC_USER" >/dev/null 2>&1 && act groupdel "$SVC_USER"
  fi
fi

step "after"
if [ "$APPLY" = yes ]; then
  say " visitorstracker units left: $("$SYSTEMCTL" list-unit-files 2>/dev/null | grep -c visitorstracker)"
  say " port 8000 still listening:  $( (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -c ':8000 ')"
  say " nginx: $("$SYSTEMCTL" is-active nginx 2>/dev/null)"
  say ""
  say " your other services:"
  "$SYSTEMCTL" list-units --type=service --state=running --no-legend --no-pager 2>/dev/null \
    | awk '{print "   " $1}' | grep -v visitorstracker
  say ""
  if [ "$SAVED" = yes ]; then
    say " Saved beforehand: $TARBALL"
    say " The app's data is in that tarball only -- delete it when you are"
    say " sure you do not want the counting history."
  else
    say " There was nothing to save, so no tarball was written."
  fi
  say " Nothing outside the list above was touched."
else
  say " dry run finished. Add --yes to carry this out."
fi
