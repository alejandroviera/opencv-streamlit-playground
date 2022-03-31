upstream ws-backend {
  server 127.0.0.1:8501;
}

server {
  listen 443 ssl;
  server_name alejandroviera.com computervision.alejandroviera.com;
  
  ssl_certificate	/home/ubuntu/ssl_certificates/cert.pem;
  ssl_certificate_key	/home/ubuntu/ssl_certificates/privkey.pem;
  ssl_protocols		TLSv1.2;
  ssl_ciphers		HIGH:!aNULL:!MD5;

  client_max_body_size 100M;
  
  location / {
    proxy_pass http://ws-backend;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header Host $http_host;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_redirect off;
    proxy_http_version 1.1;
  }
}
