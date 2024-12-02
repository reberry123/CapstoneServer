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
from datetime import datetime, timedelta
import astropy.units as u
import asyncio
import json
import time
import warnings
import numpy as np

# 경고 무시
warnings.filterwarnings('ignore', message='ERFA function "pmsafe" yielded')
lock = asyncio.Lock()
Simbad.TIMEOUT = 120

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
    distance: float

class Constellation(BaseModel):
    name: str
    nameUnicode: str
    type: str
    img: str
    ra: str
    dec: str
    alt: float
    az: float
    distance: float
    stars: List[Star]
    lines: List[List[int]]

class StellarObject(BaseModel):
    name: str
    nameUnicode: str
    id: int
    type: str
    img: str
    ra: str
    dec: str
    alt: float
    az: float
    radius: float
    flux_v: float
    distance: float

# Global state
class GlobalState:
    def __init__(self):
        self.location = Location(lat=37.5665, lon=126.9780)
        self.time: str
        self.constellations: List[Constellation] = []
        self.stellar_objs: List[StellarObject] = []
        self.sun_positions: List[Dict] = []

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
    get_sun_data()
    
    async def update_data():
        try:
            async with lock:
                for obj in global_state.stellar_objs:
                    print(f"Updating... {obj.name}")
                    calculate_obj_altaz(obj, global_state.location)
                    await asyncio.sleep(1)

                for constellation in global_state.constellations:
                    print(f"Updating... {constellation.name}")
                    alt_sum = 0
                    az_sum = 0
                    for star in constellation.stars:
                        calculate_star_altaz(star, global_state.location)
                        alt_sum += star.alt
                        az_sum += star.az
                    alt_center = alt_sum / len(constellation.stars)
                    az_center = az_sum / len(constellation.stars)
                    constellation.alt = alt_center
                    constellation.az = az_center
                    
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
        await websocket.send_text(json.dumps(global_state.sun_positions, ensure_ascii=False))

        while True:
            item = []
            for obj in global_state.stellar_objs:
                item.append({
                    "name": obj.name,
                    "type": obj.type,
                    "alt": obj.alt,
                    "az": obj.az,
                    "flux_v": obj.flux_v,
                    "distance": obj.distance
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
                    "type": constellation.type,
                    "alt": constellation.alt,
                    "az": constellation.az,
                    "distance": constellation.distance,
                    "stars": [{
                        "name": star.name,
                        "alt": star.alt,
                        "az": star.az,
                        "flux_v": star.flux_v,
                        "distance": star.distance
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
        "title": "Capstone Server API",
        "url": "/api/stellar",
        "url_by_name": "/api/stellar/{name}",
        "object": {
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
    for obj in global_state.stellar_objs:
        if obj.name == name or obj.name == name.lower():
            return obj
        
    for constellation in global_state.constellations:
        if constellation.name == name or constellation.name == name.lower():
            return constellation
        for star in constellation.stars:
            if star.name == name or star.name == name.lower():
                return star

    raise HTTPException(status_code=404, detail="Object not found")

def get_sun_data(step='1h'):
    start_date = datetime.now().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    location = {
        'lon': global_state.location.lon * u.deg,
        'lat': global_state.location.lat * u.deg,
        'elevation': 0 * u.m
    }

    observer = Horizons(id='10', location=location, epochs={
        'start': start_date,
        "stop": end_date,
        'step': step
    })

    result = observer.ephemerides()

    sun_positions = [
        {
            "alt": float(row['EL']),
            "az": float(row['AZ']),
        } for row in result
    ]

    global_state.sun_positions = sun_positions

def search_constellation(name: str):
    for constellation in global_state.constellations:
        if constellation.name == name:
            return constellation
    return None

def get_constellation_data(constellations: List[Dict]):
    for constellation in constellations:
        name = constellation['name']
        nameUnicode = constellation['nameUnicode']
        img = constellation['img']
        stars, center, distance = get_star_data(constellation['stars'])
        lines =constellation['lines']

        ra_hms = degrees_to_hms(center[0])
        dec_dms = degrees_to_dms(center[1])
        ra = f"{ra_hms[0]} {ra_hms[1]} {ra_hms[2]:.2f}"
        dec = f"{dec_dms[0]} {dec_dms[1]} {dec_dms[2]:.2f}"

        new_constellation = Constellation(name=name.replace(" ", "").lower(), nameUnicode=nameUnicode, type="constellation", img=img, ra=ra, dec=dec, alt=0, az=0, distance=distance, stars=stars, lines=lines)
        global_state.constellations.append(new_constellation)

def get_horizons_data(horizons: List[Dict]) -> List[StellarObject]:
    stellar_list = []
    obs_time = Time.now()
    for item in horizons:
        obj = Horizons(id=item["id"], location='500@399', epochs=obs_time.jd)
        eph = obj.ephemerides()

        name = item["name"]
        nameUnicode = item["nameUnicode"]
        img = f"https://port-0-capstoneserver-m2qhwewx334fe436.sel4.cloudtype.app/static/{name}.png"
        ra_hms = degrees_to_hms(eph['RA'][0])
        dec_dms = degrees_to_dms(eph['DEC'][0])
        ra = f"{ra_hms[0]} {ra_hms[1]} {ra_hms[2]:.2f}"
        dec = f"{dec_dms[0]} {dec_dms[1]} {dec_dms[2]:.2f}"
        distance = eph['delta'][0]
        radius = item['radius']
        mag = eph['V'][0]

        new_obj = StellarObject(name=name, nameUnicode=nameUnicode, id=item["id"], type=item["type"], img=img, ra=ra, dec=dec, alt=0, az=0, radius=radius, flux_v=mag, distance=distance)
        stellar_list.append(new_obj)
    return stellar_list

def get_star_data(stars: List[str]):
    simbad = Simbad()
    simbad.add_votable_fields('ids', 'flux(V)', 'pmra', 'pmdec', 'plx', 'rv_value')
    star_list = []
    coords = []
    ra_center = 0
    dec_center = 0
    retry = 5
    wait = 60
    distance_mean = 0

    for attempt in range(retry):
        try:
            result_table = simbad.query_objects(stars)
            break
        except TimeoutError:
            print(f"Attempt {attempt + 1} failed. Retrying in {wait} seconds...")
            time.sleep(wait)

    for star_name, ids, ra, dec, pm_ra_cosdec, pm_dec, parallax, radial_velocity, flux_v in zip(result_table['MAIN_ID'], result_table['IDS'], result_table['RA'], result_table['DEC'], result_table['PMRA'], result_table['PMDEC'], result_table['PLX_VALUE'], result_table['RV_VALUE'], result_table['FLUX_V']):
        if np.ma.is_masked(flux_v):
            flux_v = flux_v.filled(0)

        for alias in ids.split('|'):
            if alias.strip().startswith('NAME '):
                star_name = alias.strip().replace("NAME ", "").lower()
        coords.append((ra, dec))

        distance = calc_distance(parallax)
        distance_mean += distance

        new_star = Star(name=star_name, type="star", ra=ra, dec=dec, pm_ra_cosdec=pm_ra_cosdec, pm_dec=pm_dec, parallax=parallax, radial_velocity=radial_velocity, alt=0, az=0, flux_v = flux_v, distance=distance)
        star_list.append(new_star)

    if len(coords) > 0:
        ra_deg = [hms_to_degrees(coord[0]) for coord in coords]
        dec_deg = [dms_to_degrees(coord[1]) for coord in coords]

        ra_center = np.mean(ra_deg)
        dec_center = np.mean(dec_deg)
    distance_mean = distance_mean / len(star_list)

    return star_list, (ra_center, dec_center), distance_mean

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

def calc_distance(parallax: float) -> float:
    distance_pc = 1000 / parallax

    return distance_pc

def hms_to_degrees(ra):
    h, m, s = map(float, ra.split())
    return 15 * (h + m / 60 + s / 3600)

def dms_to_degrees(dec):
    d, m, s = map(float, dec.split())
    sign = -1 if d < 0 else 1
    return sign * (abs(d) + m / 60 + s / 3600)

def degrees_to_hms(degrees):
    h = int(degrees // 15)
    m = int((degrees % 15) * 60 / 15)
    s = (degrees % 15) * 60 % 15 * 60
    return h, m, s

def degrees_to_dms(degrees):
    d = int(degrees)
    m = int(abs(degrees - d) * 60)
    s = abs(degrees - d - m / 60) * 3600
    return d, m, s

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)