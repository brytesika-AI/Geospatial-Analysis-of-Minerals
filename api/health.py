"""GET /api/health"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from _utils import BaseHandler
class handler(BaseHandler):
    def handle_get(self, _):
        return {"status": "ok", "service": "GeoExplorer AI Africa API v2"}
