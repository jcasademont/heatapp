from app import db
from sqlalchemy.dialects.postgresql import JSON

class Temperatures(db.Model):
    __tablename__ = "temperatures"

    id = db.Column(db.Integer, primary_key=True)
    racks = db.Column(JSON)
    ahus = db.Column(JSON)

    def __init__(self, racks, ahus):
        self.racks = racks
        self.ahus = ahus

    def __repr__(self):
        return '<id {}>'.format(self.id)
