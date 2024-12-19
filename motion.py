import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import xlwings as xw

# Ezt a függvényt kell a versenyzőknek megírniuk a célnak megfelelően
def visos_control(Y, t):
    """
    Az irányításért felelős függvény.
    A versenyzők feladata, hogy az Y és t bemenetek alapján meghatározzák a napvitorlás szögét.
    Jelenleg alapértelmezetten 0 értéket ad vissza a Python kódból (Pyt=1) vagy az Excel fájlból (Pyt=0).
    """
    Pyt=1
    if Pyt==1:
        inputszog=0*np.pi/180
    else:
        # 1. Excel fájl megnyitása
        wb = xw.Book('control_excel.xlsx')  # Létrehozott Excel fájl megnyitása
        sheet = wb.sheets[0]  # Az első munkalap kiválasztása
        # 2. Adatok küldése az Excelbe
        data_to_send = [t,Y[-1,0],Y[-1,1],Y[-1,2],Y[-1,3],Y[-1,4],Y[-1,5],Y[-1,6],Y[-1,7],Y[-1,8]] #Y utolsó sora
        sheet.range('B2').value = data_to_send  # Adatok beírása az B2:K2 tartományba
        # 3. Eredmények visszaolvasása
        inputszog = sheet.range('D10').value  # A D1 tartomány eredményeinek olvasása
    return inputszog

# Segédfüggvény az okklúzió (fény blokkolásának) kiszámítására
def calculate_occlusion(observer, R, P, rP):        
    """
    Kiszámítja, hogy a napvitorla milyen mértékben van árnyékolva
    egy bolygó által a Nap fényétől.
    """
    d_R = np.linalg.norm(observer)  # Távolság a Nap (nagy gömb) és a megfigyelő között
    d_P = np.linalg.norm(observer - rP)  # Távolság a bolygó és a megfigyelő között
    d_PR = np.linalg.norm(rP)  # Távolság a Nap és a bolygó között
    fraction = 0
    if d_P < d_R:
        # Területarány számítása
        teruletszorzo = max(0, min(1, P**2 / (R**2 * (d_P / d_R)**2)))
        # Szögarány számítása (alfa, beta, gamma)
        alpha = np.arctan(R / d_R)
        beta = np.arctan(P / d_P)
        gamma = beta + np.arccos((d_R**2 + d_P**2 - d_PR**2) / (2 * d_R * d_P))
        if gamma > alpha + 2 * beta:
            szogarany = 0
        else:
            if gamma < alpha:
                szogarany = 1
            else:
                szogarany = np.interp(gamma, [0, 1], [alpha + 2 * beta, alpha])
        szogszorzo = max(0, min(1, szogarany))
        fraction = teruletszorzo * szogszorzo
    return fraction

# Globális konstansok definiálása
G = 6.67430e-11  # Gravitációs állandó
c = 3e8          # Fénysebesség
# Bolygók tömegei és sugaraik
M_sun = 1.989e30
M_venus = 4.867e24
M_earth = 5.972e24
M_moon = 7.348e22
M_mars = 6.417e23
R_sun = 696340000
R_venus = 60518000
R_earth = 6371000
R_moon = 1737400
R_mars = 3389500

AU = 149597870700  # Csillagászati egység (m)

# Napvitorlás adatai
A = 76 * 76  # Felület (m^2)
P = 4.563e-6  # Sugárzási nyomás (1 AU távolságban, N/m^2)
m = 300      # Tömeg (kg)

# `.mat` fájlok betöltése (bolygók pozíciói és pályaadatok)
venus = loadmat('venus.mat')
earth = loadmat('earth.mat')
mars = loadmat('mars.mat')
moon = loadmat('moon.mat')
# Adatok átrendezése oszlopokba
venus_data = np.column_stack((venus['x'], venus['y'], venus['z'], venus['polar']))
earth_data = np.column_stack((earth['x'], earth['y'], earth['z'], earth['polar']))
mars_data = np.column_stack((mars['x'], mars['y'], mars['z'], mars['polar']))
moon_data = np.column_stack((moon['x'], moon['y'], moon['z'], moon['polar']))

# Szimulációs kezdeti feltételek
day0 = 110  # Kezdő nap (naptári nap a szimulációban)
kiloszog = -10 * np.pi / 180  # Kilövési szög radiánban
kiloseb = 1000  # Kilövési sebesség (m/s)
kiloseb2 = -1000  # Forgási sebességből származó korrekció

# Kezdő helyzet és sebesség beállítása a Föld adatai alapján
r0 = earth_data[day0-1, :2]  # Föld pozíciója a kezdő napon
v_forg = kiloseb2 * (r0 / np.linalg.norm(r0)) @ np.array([
    [np.cos(kiloszog), -np.sin(kiloszog)],
    [np.sin(kiloszog), np.cos(kiloszog)]
])
v_kiloves = kiloseb * r0 / np.linalg.norm(r0)
v0 = ((earth_data[day0, :2] - earth_data[day0-1, :2]) / (24 * 3600)) + v_forg + v_kiloves
Th0 = np.arctan2(r0[1], r0[0])  # Kezdő szög

# Szimulációs állapot inicializálása
y = np.hstack((r0, v0))  # Kezdő pozíció és sebesség
Th = Th0
alpha = 0  # Napvitorlás szöge
Y = np.zeros((0, 9))
Y = np.vstack([Y, np.concatenate([[(day0-1) * 24 * 3600], r0, v0, [Th], [alpha], earth_data[day0-1, :2]])])

# Szimulációs iterációk: napvitorlás pályaszámítása
day = 181  # Szimuláció hossza napokban
tstep = 3600  # Időlépés másodpercben

# Számítások egy for-ciklusban (időléptetéssel)
# Szimulációs iterációk: napvitorlás pályaszámítása
day = 181  # Szimuláció hossza napokban
tstep = 3600  # Időlépés másodpercben

for t in range(day0 * 24 * 3600 + tstep, (day0 + day) * 24 * 3600 + tstep, tstep):
    # Az utolsó állapot kiolvasása
    last_element = Y[-1]
    r = last_element[1:3]  # Napvitorlás aktuális pozíciója (x, y) [m]
    v = last_element[3:5]  # Napvitorlás aktuális sebessége (vx, vy) [m/s]
    Th = last_element[5]  # Napvitorlás pálya menti szöge [rad]
    alpha = last_element[6]

    # Bolygók pozíciójának frissítése a megadott időpontra (interpolációval)
    r_venus = np.array([
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), venus_data[:, 0])(t),
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), venus_data[:, 1])(t)
    ])  # Vénusz pozíciója (x, y) [m]
    r_earth = np.array([
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), earth_data[:, 0])(t),
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), earth_data[:, 1])(t)
    ])  # Föld pozíciója (x, y) [m]
    r_moon = np.array([
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), moon_data[:, 0])(t),
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), moon_data[:, 1])(t)
    ])  # Hold pozíciója (x, y) [m]
    r_mars = np.array([
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), mars_data[:, 0])(t),
        interp1d(np.arange(0, 365*24*3600 + 1, 24*3600), mars_data[:, 1])(t)
    ])  # Mars pozíciója (x, y) [m]

    # Gravitációs gyorsulások számítása
    a_sun = (-G * M_sun / np.linalg.norm(r)**3) * r  # Nap gravitációs gyorsulása [m/s^2]
    a_venus = -G * M_venus / np.linalg.norm(r - r_venus)**3 * (r - r_venus)  # Vénusz hatása [m/s^2]
    a_earth = -G * M_earth / np.linalg.norm(r - r_earth)**3 * (r - r_earth)  # Föld hatása [m/s^2]
    a_moon = -G * M_moon / np.linalg.norm(r - r_moon)**3 * (r - r_moon)  # Hold hatása [m/s^2]
    a_mars = -G * M_mars / np.linalg.norm(r - r_mars)**3 * (r - r_mars)  # Mars hatása [m/s^2]

    # Fény blokkolásának (okklúzió) hatása
    blocking_effect_venus = calculate_occlusion(r, R_sun, R_venus, r_venus.T)
    blocking_effect_earth = calculate_occlusion(r, R_sun, R_earth, r_earth.T)
    blocking_effect_mars = calculate_occlusion(r, R_sun, R_mars, r_mars.T)
    blocking_effect_moon = calculate_occlusion(r, R_sun, R_moon, r_moon.T)

    # A legnagyobb blokkoló hatás figyelembevétele
    blocking_effect = 1 - max(blocking_effect_venus, blocking_effect_earth, blocking_effect_mars, blocking_effect_moon)

    # Az irányításhoz tartozó szög meghatározása (versenyzők által definiált)
    inputszog = visos_control(Y, t)

    # Gaussian zaj hozzáadása a vezérlési szöghöz
    sigma1 = 0.05  # Zaj szórása [rad]
    inputszog += np.random.normal(0, sigma1)

    # Szögváltozás simítása és korlátozása
    smooth = np.pi / 180  # Legnagyobb szögváltozás mértéke egy időlépés alatt [rad]
    #alpha_ = np.clip(alpha - smooth, inputszog, alpha + smooth)
    alpha_ = np.clip(inputszog, alpha - smooth, alpha + smooth)
    alpha_ = np.clip(alpha_, -np.pi/2, np.pi/2)  # Szög korlátozása [-90°, +90°]
    
    # Napvitorla normálvektorának kiszámítása
    n = np.array([
        np.cos(alpha_)*r[0] - np.sin(alpha_)*r[1],
        np.sin(alpha_)*r[0] + np.cos(alpha_)*r[1]
    ]) / np.linalg.norm(r)  # Egységvektor [m]

    # Napfény sugárzási nyomásából adódó gyorsulás
    sigma2 = 0.1  # Zaj szórása a nyomáshoz [dimenzió nélküli]
    a_radiation = P * A * np.cos(alpha_) * (1 + np.random.normal(0, sigma2)) * blocking_effect * n / m  # [m/s^2]

    # Teljes gyorsulás kiszámítása
    a_total = a_sun + a_venus + a_earth + a_moon + a_mars + a_radiation  # [m/s^2]

    # Napvitorla mozgásának előrejelzése
    v_ = v + a_total * tstep  # Új sebesség [m/s]
    r_ = r + v * tstep + 0.5 * a_total * tstep**2  # Új pozíció [m]

    # Pályaszög frissítése
    TH = np.arctan2(r[1], r[0])  # Pályaszög (azimutális) [rad]
    if TH < 0:
        TH += 2 * np.pi
    Th_ = TH + alpha_  # Szög elmozdulása a pálya mentén [rad]

    # Állapotvektor frissítése és tárolása
    Y_bovul = np.concatenate([[t], r_, v_, [Th_], [alpha_], r_earth])
    Y = np.vstack([Y, Y_bovul])

# Eredmények megjelenítése grafikonokon

# Szögváltozás (Th és alpha) időben
plt.figure()
plt.plot(Y[1:, 0], Y[1:, 5] * 180 / np.pi, label='Th')  # Napvitorlás pálya menti szöge [fok]
plt.plot(Y[1:, 0], Y[1:, 6] * 180 / np.pi, label='alpha')  # Irányítási szög [fok]
plt.xlabel("Idő [s]")  # Időtartam másodpercben
plt.ylabel("Szög [fok]")  # Szögek fokban
plt.legend()
plt.title("Napvitorlás szögének változása időben")
plt.grid()

# Napvitorlás és a Föld pályájának 2D ábrázolása
plt.figure()
plt.plot(Y[1:, 1], Y[1:, 2], label='Napvitorlás')  # Napvitorlás pályája (x, y) [m]
plt.plot(Y[1:, 7], Y[1:, 8], label='Föld')  # Föld pályája (x, y) [m]
plt.xlabel("X pozíció [m]")  # X-koordináta méterben
plt.ylabel("Y pozíció [m]")  # Y-koordináta méterben
plt.legend()
plt.title("Napvitorlás és a Föld pályája")
plt.axis('equal')  # Egyenlő tengelyarány a pontos ábrázolásért
plt.grid()

# Napvitorlás távolsága a Földtől időben
distance = np.sqrt((Y[1:, 1] - Y[1:, 7])**2 + (Y[1:, 2] - Y[1:, 8])**2)  # Távolság kiszámítása [m]
plt.figure()
plt.plot(Y[1:, 0], distance, label='Távolság a Földtől')  # Távolság változása az idő függvényében
plt.xlabel("Idő [s]")  # Időtartam másodpercben
plt.ylabel("Távolság [m]")  # Távolság méterben
plt.legend()
plt.title("Napvitorlás távolsága a Földtől időben")
plt.grid()

# Napvitorlás és Föld 3D pályája időben
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(Y[1:, 1], Y[1:, 2], Y[1:, 0], label='Napvitorlás')  # Napvitorlás pályája (x, y, idő) [m, s]
ax.plot3D(Y[1:, 7], Y[1:, 8], Y[1:, 0], label='Föld')  # Föld pályája (x, y, idő) [m, s]
ax.set_xlabel('X pozíció [m]')  # X-koordináta méterben
ax.set_ylabel('Y pozíció [m]')  # Y-koordináta méterben
ax.set_zlabel('Idő [s]')  # Idő másodpercben
ax.legend()
ax.set_title("Napvitorlás és a Föld 3D pályája")
#plt.show()

#Szögváltozás időben: Ez az ábra a napvitorlás pálya menti szögét (Th) és az irányítási szöget (alpha) mutatja fokban. Segít megérteni, hogyan módosul a napvitorlás iránya a szimuláció során.
#2D pályák: A napvitorlás és a Föld pályáját vetíti ki a 2D síkra. Ez egy vizuális áttekintést ad arról, hogy a napvitorlás hogyan mozog a Földhöz képest.
#Távolság a Földtől időben: Ez az ábra a napvitorlás és a Föld közötti távolságot ábrázolja az idő függvényében. A versenyzők célja, hogy a távolságot minimalizálják a szimuláció végére.
#3D pályák időben: Ez az ábra három dimenzióban mutatja be a napvitorlás és a Föld pályáját, időtengellyel kiegészítve. Különösen hasznos a mozgás dinamikájának megértésére.


# 2D animáció a napvitorlás és a Föld pályájáról

fig, ax = plt.subplots()
ax.set_title("Napvitorlás és Föld pályája (2D animáció)")
ax.set_xlabel("X pozíció [m]")
ax.set_ylabel("Y pozíció [m]")
ax.axis('equal')
ax.grid()

# Teljes pályák kirajzolása háttérként
ax.plot(Y[:, 7], Y[:, 8], 'g--', label='Föld pályája')  # Föld teljes pályája (zöld szaggatott)
line, = ax.plot([], [], 'b-', label='Napvitorlás pályája')  # Napvitorlás pályája (kék vonal)
point, = ax.plot([], [], 'ro', label='Napvitorlás aktuális pozíciója')  # Napvitorlás aktuális pozíciója (piros pont)

earth_point, = ax.plot([], [], 'go', label='Föld aktuális pozíciója')  # Föld aktuális pozíciója (zöld pont)
ax.legend()

# Időlépések száma az animációban
frames = len(Y)  # Az animáció lépéseinek száma
interval = 5000 / frames  # Időlépés hossza (5 másodperc alatt fut le)

def update(frame):
    """
    Frissíti az animáció adatait minden lépésben.
    """
    # Napvitorlás aktuális és eddigi pályája
    line.set_data(Y[:frame+1, 1], Y[:frame+1, 2])  # Eddigi pálya (kék vonal)
    point.set_data([Y[frame, 1]], [Y[frame, 2]])  # Aktuális pozíció (piros pont)
    #return line, point
    # Föld aktuális pozíciójának frissítése
    earth_point.set_data([Y[frame, 7]], [Y[frame, 8]])  # Föld aktuális pozíciója (zöld pont)
    return line, point, earth_point

# Animáció létrehozása
ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

# Az animáció megjelenítése
plt.show()

