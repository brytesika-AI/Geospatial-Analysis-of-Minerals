"""GET /api/health"""
from _utils import BaseHandler
class handler(BaseHandler):
    def handle_get(self, _):
        return {"status": "ok", "service": "GeoExplorer AI Africa API v2"}
