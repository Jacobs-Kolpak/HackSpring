#!/bin/sh
set -e

SSL_CERT="/etc/nginx/ssl/fullchain.crt"
SSL_KEY="/etc/nginx/ssl/private.key"
CONF_DIR="/etc/nginx/conf.d"

if [ -f "$SSL_CERT" ] && [ -f "$SSL_KEY" ]; then
    echo "[entrypoint] SSL-сертификаты найдены -> запуск с HTTPS + HTTP"
    cp /etc/nginx/templates/nginx.conf "$CONF_DIR/default.conf"
else
    echo "[entrypoint] SSL-сертификаты НЕ найдены -> запуск только HTTP (порт 80)"
    cp /etc/nginx/templates/nginx-no-ssl.conf "$CONF_DIR/default.conf"
fi

exec nginx -g 'daemon off;'
