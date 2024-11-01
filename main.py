from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi_utils.tasks import repeat_every
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Dict, Any, Optional
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
    stars: List[Star]
    lines: List[List[int]]

# Global state
class GlobalState:
    def __init__(self):
        self.location = Location(lat=37.5665, lon=126.9780)
        self.time: str
        self.constellations: List[Constellation] = []

global_state = GlobalState()

def process_data(constellations: List[Dict]):
    try:
        get_constellation_data(constellations)
        print("Loaded constellations!")
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

# FastAPI application setup
app = FastAPI()

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
    process_data(constellations)

    async def update_data():
        try:
            async with lock:
                i = 1
                m = len(global_state.constellations)
                for constellation in global_state.constellations:
                    print(f"\r({int((i/m)*100)}%) Updating {constellation.name}          ", end="")
                    for star in constellation.stars:
                        calculate_altaz(star, global_state.location)
                    i += 1
                    await asyncio.sleep(1)
                print("\r", end="")
                print("Updated successfully!     ")
                
        except Exception as e:
            print(f"Error updating data: {e}")

    task = asyncio.create_task(
        repeat_every(seconds=120)(update_data)()
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
        data = await websocket.receive_text()
        print('Received data:', data)
        
        location_data = json.loads(data)
        new_location = Location(**location_data["location"])
        global_state.location = new_location
        print(f"Updated location: {global_state.location}")

        while True:
            send_list = []
            for constellation in global_state.constellations:
                send_list.append(constellation)
                if len(send_list) >= 15:
                    try:
                        await websocket.send_text(json.dumps(send_list, ensure_ascii=False))
                        print(f"Sent constellation batch with {len(send_list)} constellations")
                        send_list = []
                    except Exception as e:
                        print(f"Error sending data: {e}")
                        raise
                await asyncio.sleep(1)
            try:
                await websocket.send_text(json.dumps(send_list, ensure_ascii=False))
                print(f"Sent constellation batch with {len(send_list)} constellations")
            except Exception as e:
                print(f"Error sending data: {e}")
                raise

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
        "url": "/api/constellations",
        "star": {
            "ra": "적경 Right Ascension",
            "dec": "적위 Declination",
            "alt": "고도 Altitude",
            "az": "방위각 Azimuth",
            "flux_v": "겉보기 등급 Flux V"
        }
    }

@app.get("/api/constellations")
async def get_constellations():
    return global_state.constellations

@app.get("/api/constellations/{name}")
async def get_constellations_by_name(name: str):
    for constellation in global_state.constellations:
        if constellation.name == name:
            return constellation
    
    return HTTPException(status_code=404, detail="Constellation not found")

@app.get("/api/horizons/")
async def get_horizons():
    return {
        "message": "umm"
    }

def search_constellation(name: str):
    for constellation in global_state.constellations:
        if constellation.name == name:
            return constellation
    return None

def get_constellation_data(constellations: List[Dict]):
    for i in range(len(constellations)):
        constellation = constellations[i]
        name = constellation['name']
        nameUnicode = constellation['nameUnicode']
        stars = get_star_data(constellation['stars'])
        lines =constellation['lines']
        
        new_constellation = Constellation(name=name, nameUnicode=nameUnicode, stars=stars, lines=lines)
        
        global_state.constellations.append(new_constellation)

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

        new_star = Star(name=star_name, ra=ra, dec=dec, pm_ra_cosdec=pm_ra_cosdec, pm_dec=pm_dec, parallax=parallax, radial_velocity=radial_velocity, alt=0, az=0, flux_v = flux_v)
        star_list.append(new_star)

    return star_list

def calculate_altaz(star: Star, location: Location):
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

# 태양계 내부 천체 검색 (JPL HORIZONS) --- 테스트용
def get_planet_data(planet_id, obs_loc):
    obs_location = {'lon': obs_loc[1] * u.deg,  # 관측 위치 설정
                     'lat': obs_loc[0] * u.deg,
                     'elevation': 0 * u.m}
    # obs_time = Time.now()                     # 관측 시간 설정
    obs_time = Time("2024-10-08 9:00:00")

    # 천체 검색
    planet = Horizons(id=planet_id, location=obs_location, epochs=obs_time.jd)
    planet_ephem = planet.ephemerides()

    # 정보 추출
    target_name = planet_ephem['targetname'][0]
    ra = planet_ephem['RA'][0]      # 적경 [deg]
    dec = planet_ephem['DEC'][0]    # 적위 [deg]
    alt = planet_ephem['EL'][0]     # 고도 [deg]
    az = planet_ephem['AZ'][0]      # 방위각 [deg]
    v = planet_ephem['V'][0]        # 겉보기 등급

    print(f'TIME {obs_time}')
    print(f'TARGET {target_name}')
    print(f'RA {ra}, DEC {dec}, ALT {alt}, AZ {az}')
    print(f'V {v}')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)