IMAGE ?= eopf-geozarr:dev
WF ?= geozarr-convert.yaml
CLUSTER ?= k3s-default   # change if your k3d cluster has a different name

.PHONY: build load-k3d load-minikube argo-install argo-ui-dev submit status logs latest clean

build:
	docker build -t $(IMAGE) .

# k3d: import local image into cluster's containerd
load-k3d:
	k3d image import $(IMAGE) --cluster $(CLUSTER) || \
	  (docker save $(IMAGE) | docker exec -i $$(docker ps --format '{{.Names}}' | grep $(CLUSTER)-server-0) ctr -n k8s.io images import -)

# minikube: build inside minikube's docker
load-minikube:
	eval "$$(minikube docker-env)"; docker build -t $(IMAGE) .

# install Argo Workflows (3.7.1) if missing
argo-install:
	kubectl create ns argo 2>/dev/null || true
	kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.7.1/install.yaml
	kubectl -n argo rollout status deploy/workflow-controller
	kubectl -n argo rollout status deploy/argo-server

# dev UI: HTTP, no token
argo-ui-dev:
	kubectl -n argo patch deploy argo-server --type='json' -p='[ \
	  {"op":"replace","path":"/spec/template/spec/containers/0/args", \
	   "value":["server","--auth-mode=server","--secure=false"]} ]' || true
	kubectl -n argo rollout status deploy/argo-server
	kubectl -n argo port-forward svc/argo-server 2746:2746

submit:
	argo submit -n argo $(WF) --watch

status:
	argo list -n argo || true
	kubectl -n argo get wf || true
	kubectl -n argo get pods || true

logs:
	argo logs -n argo @latest -f

latest:
	argo get -n argo @latest

clean:
	argo delete -n argo --all || true
	kubectl -n argo delete pod -l workflows.argoproj.io/completed=true --force --grace-period=0 || true
