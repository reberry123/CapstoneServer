from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    alt: float
    az: float
    fluxV: float

class Constellation(BaseModel):
    name: str
    stars: List[Star]
    lines: List[List[int]]

class ServerData(BaseModel):
    location: Location
    time: str
    constellations: List[Constellation]

# Global state
class GlobalState:
    def __init__(self):
        self.location = Location(lat=37.5665, lon=126.9780)
        self.result: List[Dict[str, Any]] = [{} for _ in range(6)]
        self.data_index = 0
        self.batch_index = 0

global_state = GlobalState()

# Main data processing
async def process_data(parsed_data: List[Dict], location: Location):
    # async with global_state.processing_lock:
    cst: List[Dict] = []
    
    try:
        for i in range(global_state.data_index, len(parsed_data)):
            obj = parsed_data[i]
            cst_name = obj['name']
            cst_nameUnicode = obj['nameUnicode']
            cst_data = get_star_datas(obj['stars'], location)
            cst_line = obj['lines']
            
            new_cst = {
                'name': cst_name,
                'nameUnicode': cst_nameUnicode,
                'stars': cst_data,
                'lines': cst_line
            }
            cst.append(new_cst)
            global_state.data_index += 1
            print(f'{global_state.batch_index * 15 + global_state.data_index}/88 Updated')

            if len(cst) >= 15:
                server_data = {
                    'location': location.model_dump(),
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'constellations': cst
                }
                global_state.result[global_state.batch_index] = server_data
                global_state.batch_index += 1
                if global_state.batch_index >= len(global_state.result):
                    global_state.batch_index = 0

                break

        if global_state.data_index >= len(parsed_data):
            global_state.data_index = 0
        print('Complete!')
        
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
    parsed_data = parse_constellations_data()

    async def update_data():
        try:
            await process_data(parsed_data, global_state.location)
            print("Data updated successfully!")
        except Exception as e:
            print(f"Error updating data: {e}")

    task = asyncio.create_task(
        repeat_every(seconds=120)(update_data)()
    )
    
    yield
    
    # Shutdown
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
        # Handle initial location data
        data = await websocket.receive_text()
        print('Received data:', data)
        
        location_data = json.loads(data)
        new_location = Location(**location_data["location"])
        global_state.location = new_location
        print(f"Updated location: {global_state.location}")

        #초기 데이터 전송
        for item in global_state.result:
            if item:
                try:
                    await websocket.send_text(json.dumps(item, ensure_ascii=False))
                    print(f"Sent constellation batch with {len(item.get('constellations', []))} constellations")
                except Exception as e:
                    print(f"Error sending data: {e}")
                    raise
            await asyncio.sleep(1)

        # 메인 웹소켓 루프
        while True:
            await asyncio.sleep(180)
            for item in global_state.result:
                if item:
                    try:
                        await websocket.send_text(json.dumps(item, ensure_ascii=False))
                        print(f"Sent constellation batch with {len(item.get('constellations', []))} constellations")
                    except Exception as e:
                        print(f"Error sending data: {e}")
                        raise
                await asyncio.sleep(1)

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
        "message": "cibal"
    }

@app.get("/api/constellations/{name}")
async def get_constellations(name: str = None, lat: float = None, lon: float = None):
    constellation = search_constellation(name)

    return {
        "name": constellation['name'],
        "nameUnicode": constellation['nameUnicode'],
        "ra": constellation['ra'],
        "dec": constellation['dec'],
        "alt": constellation['alt'],
        "az": constellation['az'],
        "flux_v": constellation['flux_v']
    }


def search_constellation(name: str):
    for item in global_state.result:
        for constellation in item:
            if constellation['name'] == name:
                return constellation

# Utility functions
def get_star_datas(stars: List[str], location: Location) -> List[Dict]:
    simbad = Simbad()
    simbad.add_votable_fields('flux(V)', 'pmra', 'pmdec', 'plx', 'rv_value')
    obs_location = EarthLocation(lat=location.lat * u.deg, lon=location.lon * u.deg, height=0 * u.m)
    obs_time = Time.now()
    star_datas = []
    retry = 5
    wait = 60
    i = 0

    for attempt in range(retry):
        try:
            result_table = simbad.query_objects(stars)
            break
        except TimeoutError:
            print(f"Attempt {attempt + 1} failed. Retrying in {wait} seconds...")
            time.sleep(wait)
    

    for star_name, ra, dec, pm_ra_cosdec, pm_dec, parallax, radial_velocity, flux_v in zip(result_table['MAIN_ID'], result_table['RA'], result_table['DEC'], result_table['PMRA'], result_table['PMDEC'], result_table['PLX_VALUE'], result_table['RV_VALUE'], result_table['FLUX_V']):

        # SkyCoord 객체로 변환
        star_coord = SkyCoord(
            ra=ra,
            dec=dec,
            unit=(u.hourangle, u.deg),
            frame='icrs',
            pm_ra_cosdec=pm_ra_cosdec * u.mas/u.yr,
            pm_dec=pm_dec * u.mas/u.yr,
            distance=1000/parallax * u.parsec,
            radial_velocity=radial_velocity * u.km/u.s,
            obstime=Time('2000-01-01T00:00:00')
        )

        star_now = star_coord.apply_space_motion(new_obstime=obs_time)
        altaz_frame = AltAz(obstime=obs_time, location=obs_location)
        star_altaz = star_now.transform_to(altaz_frame)
        if np.ma.is_masked(flux_v):
            flux_v = flux_v.filled(np.nan)

        star_data = {
            'id': star_name,
            'ra': star_now.ra.degree,
            'dec': star_now.dec.degree,
            'alt': star_altaz.alt.degree,
            'az': star_altaz.az.degree,
            'flux_v': float(flux_v),
        }

        star_datas.append(star_data)
        i += 1

    return star_datas

def parse_constellations_data() -> List[Dict]:
    with open('test.json', 'r', encoding="UTF8") as f:
        data = json.load(f)
    return data

# 태양계 외부 천체 검색 (SIMBAD)
def get_star_datas_test(star_names, obs_loc):
    simbad = Simbad()
    simbad.add_votable_fields('flux(V)', 'pmra', 'pmdec', 'plx', 'rv_value')
    result_table = simbad.query_objects(star_names)
    obs_location = EarthLocation(lat=obs_loc["lat"] * u.deg, lon=obs_loc["lon"] * u.deg, height=0 * u.m)
    obs_time = Time.now()
    star_datas = []
    i = 0

    for star_name, ra, dec, pm_ra_cosdec, pm_dec, parallax, radial_velocity, flux_v in zip(result_table['MAIN_ID'], result_table['RA'], result_table['DEC'], result_table['PMRA'], result_table['PMDEC'], result_table['PLX_VALUE'], result_table['RV_VALUE'], result_table['FLUX_V']):

        # SkyCoord 객체로 변환
        star_coord = SkyCoord(
            ra=ra,
            dec=dec,
            unit=(u.hourangle, u.deg),
            frame='icrs',
            pm_ra_cosdec=pm_ra_cosdec * u.mas/u.yr,
            pm_dec=pm_dec * u.mas/u.yr,
            distance=1000/parallax * u.parsec,
            radial_velocity=radial_velocity * u.km/u.s,
            obstime=Time('2000-01-01T00:00:00')
        )

        star_now = star_coord.apply_space_motion(new_obstime=obs_time)
        altaz_frame = AltAz(obstime=obs_time, location=obs_location)
        star_altaz = star_now.transform_to(altaz_frame)
        if np.ma.is_masked(flux_v):
            flux_v = flux_v.filled(np.nan)

        star_data = {
            'id': star_name,
            'ra': star_now.ra.degree,
            'dec': star_now.dec.degree,
            'alt': star_altaz.alt.degree,
            'az': star_altaz.az.degree,
            'flux_v': float(flux_v),
        }

        star_datas.append(star_data)
        i += 1

    return star_datas

# 태양계 외부 단일 천체 검색 (SIMBAD) --- 테스트용
def get_star_data(star_name):
    # 별의 위치 정보와 고유 운동 데이터 가져오기
    simbad = Simbad()
    simbad.add_votable_fields('pmra', 'pmdec', 'plx', 'rv_value')
    result_table = simbad.query_object("Betelgeuse")

    # 정보 추출
    name = result_table['NAME'][0]
    ra = result_table['RA'][0]
    dec = result_table['DEC'][0]
    pm_ra_cosdec = result_table['PMRA'][0]
    pm_dec = result_table['PMDEC'][0]
    parallax = result_table['PLX_VALUE'][0]
    radial_velocity = result_table['RV_VALUE'][0]

    # SkyCoord 객체로 변환
    star_coord = SkyCoord(
        ra=ra,
        dec=dec,
        unit=(u.hourangle, u.deg),
        frame='icrs',
        pm_ra_cosdec=pm_ra_cosdec * u.mas/u.yr,
        pm_dec=pm_dec * u.mas/u.yr,
        distance=1000/parallax * u.parsec,
        radial_velocity=radial_velocity * u.km/u.s,
        obstime=Time('2000-01-01T00:00:00')
    )

    # 관측 위치 설정
    obs_location = EarthLocation(lat=37.5665 * u.deg, lon=126.9780 * u.deg, height=0 * u.m)

    # 관측 시간 설정
    obs_time = Time("2024-10-05T15:00:00")

    # 고유 운동을 반영하여 위치 계산
    star_now = star_coord.apply_space_motion(new_obstime=obs_time)

    # 지평 좌표계로 변환
    altaz_frame = AltAz(obstime=obs_time, location=obs_location)
    star_altaz = star_now.transform_to(altaz_frame)

    # 결과 출력
    print(f"관측 위치: (위도: {obs_location.lat:.2f}, 경도: {obs_location.lon:.2f})")
    print(f"관측 시간: {obs_time.iso} UTC")
    print(f"적경 (RA): {star_now.ra:.6f}")
    print(f"적위 (Dec): {star_now.dec:.6f}")
    print(f"고도: {star_altaz.alt:.2f}")
    print(f"방위각: {star_altaz.az:.2f}")

    star_data = {
        'name': star_name,
        'ra': star_now.ra,
        'dec': star_now.dec,
        'alt': star_altaz.alt,
        'az': star_altaz.az,
    }

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