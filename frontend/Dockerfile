# syntax=docker/dockerfile:1.6
# frontend/Dockerfile —— node 构建产物交给 nginx 托管
#
# 不用 `vite preview` 起生产服务：它没有 /api 反代，也不该出现在生产链路上。
# 这里 node 阶段只负责产出 dist/，最终镜像里没有任何 node 运行时。

# ---------- Stage 1: 构建 ----------
FROM node:20-alpine AS builder

# 默认官方源；国内可传 --build-arg NPM_REGISTRY=https://registry.npmmirror.com
ARG NPM_REGISTRY="https://registry.npmjs.org"

WORKDIR /app

# 先只拷锁文件：依赖没变时这一层能命中缓存，改业务代码不必重装 node_modules
COPY package.json package-lock.json ./
RUN npm config set registry "$NPM_REGISTRY" && npm ci

COPY . .
RUN npm run build


# ---------- Stage 2: 托管 ----------
FROM nginx:1.27-alpine AS runtime

# 默认配置会去 proxy_pass 一个不存在的东西，直接换掉
RUN rm -f /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q --spider http://127.0.0.1/ || exit 1
