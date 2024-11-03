from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi_utils.tasks import repeat_every
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict
from pydantic import BaseModel, Field
from astroquery.simbad import Simbad
from astroquery.jplhorizons import Horizons
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
import asyncio
import json
import time
import copy
import warnings
import numpy as np

# 경고 무시
warnings.filterwarnings('ignore', message='ERFA function "pmsafe" yielded')
lock = asyncio.Lock()
Simbad.TIMEOUT = 120

# 예시 데이터 2
planets = [
    10,     # Sun
    199,    # Mercury
    299,    # Venus
    # 399,  # Earth
    499,    # Mars
    599,    # Jupiter
    699,    # Saturn
    799,    # Uranus
    899,    # Neptune
    999,    # Pluto

    # 301,  # Moon
    # 1,    # Ceres
    # 2,    # Pallas
    # 4,    # Vesta
    # 1P,   # Halley's Comet
    # -125544,  # International Space Station
]

# Type definitions
class Location(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)

class Star(BaseModel):
    name: str
    type: str
    ra: str
    dec: str
    pm_ra_cosdec: float
    pm_dec: float
    parallax: float
    radial_velocity: float
    alt: float
    az: float
    flux_v: float

class Constellation(BaseModel):
    name: str
    nameUnicode: str
    type: str
    stars: List[Star]
    lines: List[List[int]]

class StellarObject(BaseModel):
    name: str
    id: int
    type: str
    img: str
    ra: float
    dec: float
    alt: float
    az: float
    radius: float
    magnitude: float
    distance: float

# Global state
class GlobalState:
    def __init__(self):
        self.location = Location(lat=37.5665, lon=126.9780)
        self.time: str
        self.constellations: List[Constellation] = []
        self.stellar_objs: List[StellarObject] = []

global_state = GlobalState()

def process_data(constellations: List[Dict], objects: List[StellarObject]):
    try:
        get_constellation_data(constellations)
        global_state.stellar_objs = get_horizons_data(objects)
        print("Loaded constellations!")
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

# FastAPI application setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="img"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    constellations = parse_constellations_data()
    objects = parse_horizons_data()
    process_data(constellations, objects)
    
    async def update_data():
        try:
            async with lock:
                for obj in global_state.stellar_objs:
                    print(f"Updating... {obj.name}")
                    calculate_obj_altaz(obj, global_state.location)
                    await asyncio.sleep(1)

                for constellation in global_state.constellations:
                    print(f"Updating... {constellation.name}")
                    for star in constellation.stars:
                        calculate_star_altaz(star, global_state.location)
                    await asyncio.sleep(1)

                print("Updated successfully!")
        except Exception as e:
            print(f"Error updating data: {e}")

    task = asyncio.create_task(
        repeat_every(seconds=60)(update_data)()
    )
    
    yield
    
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass

app.router.lifespan_context = lifespan

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # data = await websocket.receive_text()
        # print('Received data:', data)
        
        # location_data = json.loads(data)
        # new_location = Location(**location_data["location"])
        # global_state.location = new_location
        # print(f"Updated location: {global_state.location}")

        while True:
            item = []
            for obj in global_state.stellar_objs:
                item.append({
                    "name": obj.name,
                    "type": obj.type,
                    "img": obj.img,
                    "ra": obj.ra,
                    "dec": obj.dec,
                    "alt": obj.alt,
                    "az": obj.az,
                    "radius": obj.radius,
                    "distance": obj.distance,
                    "flux_v": obj.magnitude
                })
            try:
                await websocket.send_text(json.dumps(item, ensure_ascii=False))
                print(f"Sent constellation batch with {len(item)} stellar objects")
                item = []
            except Exception as e:
                print(f"Error sending data: {e}")
                raise

            for constellation in global_state.constellations:
                item.append({
                    "name": constellation.name,
                    "nameUnicode": constellation.nameUnicode,
                    "type": constellation.type,
                    "stars": [{
                        "name": star.name,
                        "alt": star.alt,
                        "az": star.az,
                        "flux_v": star.flux_v
                    } for star in constellation.stars],
                    "lines": [line for line in constellation.lines]
                })
                if len(item) >= 15:
                    try:
                        await websocket.send_text(json.dumps(item, ensure_ascii=False))
                        print(f"Sent constellation batch with {len(item)} constellations")
                        item = []
                    except Exception as e:
                        print(f"Error sending data: {e}")
                        raise
            try:
                await websocket.send_text(json.dumps(item, ensure_ascii=False))
                print(f"Sent constellation batch with {len(item)} constellations")
            except Exception as e:
                print(f"Error sending data: {e}")
                raise

            await asyncio.sleep(60)

    except WebSocketDisconnect:
        print('Client disconnected')
    except Exception as e:
        print(f'WebSocket error: {e}')
    finally:
        manager.disconnect(websocket)
        await websocket.close()

@app.get("/")
async def read_root():
    return {
        "url": "/api/stellar",
        "star": {
            "ra": "적경 Right Ascension",
            "dec": "적위 Declination",
            "alt": "고도 Altitude",
            "az": "방위각 Azimuth",
            "radius": "적도 반지름(km)",
            "flux_v": "가시등급 (겉보기등급)",
            "distance": "지구와의 거리(AU)"
        }
    }

@app.get("/api/stellar")
async def get_stellar_objects():
    send_data = []
    send_data.append(global_state.stellar_objs)
    send_data.extend(global_state.constellations)
    return send_data

@app.get("/api/stellar/{name}")
async def get_stellar_objects_by_name(name: str):
    for constellation in global_state.constellations:
        if constellation.name == name or constellation.name == name.lower():
            return constellation
    
    for obj in global_state.stellar_objs:
        if obj.name == name or obj.name == name.lower():
            return obj
    
    return HTTPException(status_code=404, detail="Constellation not found")

def search_constellation(name: str):
    for constellation in global_state.constellations:
        if constellation.name == name:
            return constellation
    return None

def get_constellation_data(constellations: List[Dict]):
    for constellation in constellations:
        name = constellation['name']
        nameUnicode = constellation['nameUnicode']
        stars = get_star_data(constellation['stars'])
        lines =constellation['lines']
        
        new_constellation = Constellation(name=name, nameUnicode=nameUnicode, type="constellation", stars=stars, lines=lines)
        
        global_state.constellations.append(new_constellation)

def get_horizons_data(horizons: List[Dict]) -> List[StellarObject]:
    stellar_list = []
    obs_time = Time.now()
    for item in horizons:
        obj = Horizons(id=item["id"], location='500@399', epochs=obs_time.jd)
        eph = obj.ephemerides()

        name = item["name"]
        img = f"https://port-0-capstoneserver-m2qhwewx334fe436.sel4.cloudtype.app/static/{name}.png"
        ra = eph['RA'][0]
        dec = eph['DEC'][0]
        distance = eph['delta'][0]
        radius = item['radius']
        mag = eph['V'][0]

        new_obj = StellarObject(name=name, id=item["id"], type=item["type"], img=img, ra=ra, dec=dec, alt=0, az=0, radius=radius, magnitude=mag, distance=distance)
        stellar_list.append(new_obj)
    return stellar_list

def get_star_data(stars: List[str]) -> List[Star]:
    simbad = Simbad()
    simbad.add_votable_fields('flux(V)', 'pmra', 'pmdec', 'plx', 'rv_value')
    star_list = []
    retry = 5
    wait = 60

    for attempt in range(retry):
        try:
            result_table = simbad.query_objects(stars)
            break
        except TimeoutError:
            print(f"Attempt {attempt + 1} failed. Retrying in {wait} seconds...")
            time.sleep(wait)

    for star_name, ra, dec, pm_ra_cosdec, pm_dec, parallax, radial_velocity, flux_v in zip(result_table['MAIN_ID'], result_table['RA'], result_table['DEC'], result_table['PMRA'], result_table['PMDEC'], result_table['PLX_VALUE'], result_table['RV_VALUE'], result_table['FLUX_V']):
        if np.ma.is_masked(flux_v):
            flux_v = flux_v.filled(0)

        new_star = Star(name=star_name, type="star", ra=ra, dec=dec, pm_ra_cosdec=pm_ra_cosdec, pm_dec=pm_dec, parallax=parallax, radial_velocity=radial_velocity, alt=0, az=0, flux_v = flux_v)
        star_list.append(new_star)

    return star_list

def calculate_obj_altaz(obj: StellarObject, location: Location):
    obs_location = {
        'lon': location.lon * u.deg,
        'lat': location.lat * u.deg,
        'elevation': 0 * u.m
    }
    obs_time = Time.now()

    stellar_obj = Horizons(id=obj.id, location=obs_location, epochs=obs_time.jd)
    ephem = stellar_obj.ephemerides()

    obj.alt = ephem['EL'][0]
    obj.az = ephem['AZ'][0]

def calculate_star_altaz(star: Star, location: Location):
    obs_location = EarthLocation(lat=location.lat * u.deg, lon=location.lon * u.deg, height=0 * u.m)
    obs_time = Time.now()

    star_coord = SkyCoord(
        ra=star.ra,
        dec=star.dec,
        unit=(u.hourangle, u.deg),
        frame='icrs',
        pm_ra_cosdec=star.pm_ra_cosdec * u.mas/u.yr,
        pm_dec=star.pm_dec * u.mas/u.yr,
        distance=1000/star.parallax * u.parsec,
        radial_velocity=star.radial_velocity * u.km/u.s,
        obstime=Time('2000-01-01T00:00:00')
    )

    star_now = star_coord.apply_space_motion(new_obstime=obs_time)
    altaz_frame = AltAz(obstime=obs_time, location=obs_location)
    star_altaz = star_now.transform_to(altaz_frame)

    global_state.time = obs_time
    star.alt = star_altaz.alt.degree
    star.az = star_altaz.az.degree

def parse_constellations_data() -> List[Dict]:
    with open('constellation.json', 'r', encoding="UTF8") as f:
        data = json.load(f)
    return data

def parse_horizons_data() -> List[Dict]:
    with open('horizons.json', 'r', encoding="UTF8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # get_planet_data(10, (37.5665, 126.9780))