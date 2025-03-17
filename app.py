from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import calendar

app = Flask(__name__)
model = load_model("model.h5")

companies = [
    "GP", "00DS30", "00DSES", "00DSEX", "00DSMEX", "1JANATAMF",
    "1STPRIMFMF", "AAMRANET", "AAMRATECH", "ABB1STMF", "ABBANK", "ABBLPBOND",
    "ACFL", "ACHIASF", "ACIFORMULA", "ACI", "ACMELAB", "ACMEPL", "ACTIVEFINE",
    "ADNTEL", "ADVENT", "AFCAGRO", "AFTABAUTO", "AGNISYSL", "AGRANINS",
    "AIBL1STIMF", "AIL", "AL-HAJTEX", "ALARABANK", "ALIF", "ALLTEX",
    "AMANFEED", "AMBEEPHA", "AMCL(PRAN)", "AMPL", "ANLIMAYARN", "ANWARGALV",
    "AOL", "AOPLC", "APEXFOODS", "APEXFOOT", "APEXSPINN", "APEXTANRY",
    "APEXWEAV", "APOLOISPAT", "APSCLBOND", "ARAMITCEM", "ARAMIT",
    "ARGONDENIM", "ASIAINS", "ASIAPACINS", "ATCSLGF", "ATLASBANG",
    "AZIZPIPES", "BANGAS", "BANKASI1PB", "BANKASIA", "BARKAPOWER",
    "BATASHOE", "BATBC", "BAYLEASING", "BBSCABLES", "BBS", "BDAUTOCA",
    "BDCOM", "BDFINANCE", "BDLAMPS", "BDPAINTS", "BDTHAIFOOD", "BDTHAI",
    "BDWELDING", "BEACHHATCH", "BEACONPHAR", "BENGALBISC", "BENGALWTL",
    "BERGERPBL", "BESTHLDNG", "BEXGSUKUK", "BEXIMCO", "BGIC", "BIFC",
    "BNICL", "BPML", "BPPL", "BRACBANK", "BSCCL", "BSC", "BSRMLTD",
    "BSRMSTEEL", "BXPHARMA", "BXSYNTH", "CAPITECGBF", "CAPMBDBLMF",
    "CAPMIBBLMF", "CBLPBOND", "CENTRALINS", "CENTRALPHL", "CITYBANK",
    "CITYGENINS", "CLICL", "CNATEX", "CONFIDCEM", "CONTININS",
    "COPPERTECH", "CROWNCEMNT", "CRYSTALINS", "CVOPRL", "DACCADYE",
    "DAFODILCOM", "DBH1STMF", "DBH", "DBLPBOND", "DELTALIFE", "DELTASPINN",
    "DESCO", "DESHBANDHU", "DGIC", "DHAKABANK", "DHAKAINS", "DOMINAGE",
    "DOREENPWR", "DSHGARME", "DSSL", "DULAMIACOT", "DUTCHBANGL",
    "EASTERNINS", "EASTLAND", "EASTRNLUB", "EBL1STMF", "EBLNRBMF", "EBL",
    "ECABLES", "EGEN", "EHL", "EIL", "EMERALDOIL", "ENVOYTEX", "EPGL",
    "ESQUIRENIT", "ETL", "EXIM1STMF", "EXIMBANK", "FAMILYTEX", "FARCHEM",
    "FAREASTFIN", "FAREASTLIF", "FASFIN", "FBFIF", "FEDERALINS", "FEKDIL",
    "FINEFOODS", "FIRSTFIN", "FIRSTSBANK", "FORTUNE", "FUWANGCER",
    "FUWANGFOOD", "GBBPOWER", "GEMINISEA", "GENEXIL", "GENNEXT", "GHAIL",
    "GHCL", "GIB", "GLDNJMF", "GLOBALINS", "GOLDENSON", "GPHISPAT",
    "GQBALLPEN", "GRAMEENS2", "GREENDELMF", "GREENDELT", "GSPFINANCE",
    "HAKKANIPUL", "HEIDELBCEM", "HFL", "HIMADRI", "HRTEX", "HWAWELLTEX",
    "IBNSINA", "IBP", "ICB3RDNRB", "ICBAGRANI1", "ICBAMCL2ND", "ICBEPMF1S1",
    "ICBIBANK", "ICBSONALI1", "ICB", "ICICL", "IDLC", "IFADAUTOS",
    "IFIC1STMF", "IFIC", "IFILISLMF1", "ILFSL", "IMAMBUTTON", "INDEXAGRO",
    "INTECH", "INTRACO", "IPDC", "ISLAMIBANK", "ISLAMICFIN", "ISLAMIINS",
    "ISNLTD", "ITC", "JAMUNABANK", "JAMUNAOIL", "JANATAINS", "JHRML",
    "JMISMDL", "JUTESPINN", "KARNAPHULI", "KAY&QUE", "KBPPWBIL", "KBSEED",
    "KDSALTD", "KEYACOSMET", "KFL", "KOHINOOR", "KPCL", "KPPL", "KTL",
    "LANKABAFIN", "LEGACYFOOT", "LHBL", "LIBRAINFU", "LINDEBD", "LOVELLO",
    "LRBDL", "LRGLOBMF1", "MAKSONSPIN", "MALEKSPIN", "MAMUNAGRO",
    "MARICO", "MASTERAGRO", "MATINSPINN", "MBL1STMF", "MEGCONMILK",
    "MEGHNACEM", "MEGHNAINS", "MEGHNALIFE", "MEGHNAPET", "MERCANBANK",
    "MERCINS", "METROSPIN", "MHSML", "MIDASFIN", "MIDLANDBNK",
    "MIRACLEIND", "MIRAKHTER", "MITHUNKNIT", "MJLBD", "MKFOOTWEAR",
    "MLDYEING", "MONNOAGML", "MONNOCERA", "MONNOFABR", "MONOSPOOL",
    "MOSTFAMETL", "MPETROLEUM", "MTB", "NAHEEACP", "NATLIFEINS",
    "NAVANACNG", "NAVANAPHAR", "NBL", "NCCBANK", "NCCBLMF1", "NEWLINE",
    "NFML", "NHFIL", "NIALCO", "NITOLINS", "NORTHERN", "NORTHRNINS",
    "NPOLYMER", "NRBCBANK", "NTC", "NTLTUBES", "NURANI", "OAL", "OIMEX",
    "OLYMPIC", "ONEBANKPLC", "ORIONINFU", "ORIONPHARM", "ORYZAAGRO",
    "PADMALIFE", "PADMAOIL", "PAPERPROC", "PARAMOUNT", "PDL", "PENINSULA",
    "PEOPLESINS", "PF1STMF", "PHARMAID", "PHENIXINS", "PHOENIXFIN",
    "PHPMF1", "PIONEERINS", "POPULAR1MF", "POPULARLIF", "POWERGRID",
    "PRAGATIINS", "PRAGATILIF", "PREMIERBAN", "PREMIERCEM", "PREMIERLEA",
    "PRIME1ICBA", "PRIMEBANK", "PRIMEFIN", "PRIMEINSUR", "PRIMELIFE",
    "PRIMETEX", "PROGRESLIF", "PROVATIINS", "PTL", "PUBALIBANK",
    "PURABIGEN", "QUASEMIND", "QUEENSOUTH", "RAHIMAFOOD", "RAHIMTEXT",
    "RAKCERAMIC", "RANFOUNDRY", "RDFOOD", "RECKITTBEN", "REGENTTEX",
    "RELIANCE1", "RELIANCINS", "RENATA", "RENWICKJA", "REPUBLIC",
    "RINGSHINE", "RNSPIN", "ROBI", "RSRMSTEEL", "RUNNERAUTO", "RUPALIBANK",
    "RUPALIINS", "RUPALILIFE", "SADHESIVE", "SAFKOSPINN"
]

def predict_stock_prices(company, days=30):
    df = pd.read_csv(f'stock_data.csv')  # Ensure data exists
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X = scaled_data[-30:].reshape(1, 30, 1)
    
    predictions = []
    for _ in range(days):
        pred = model.predict(X)
        predictions.append(pred[0, 0])
        X = np.append(X[:, 1:, :], [[pred]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()




def get_calendar():
    bd_holidays = {1: [1], 2: [21], 3: [26], 4: [14], 5: [1], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [16, 25]}
    year, month = 2025, 3  # Example: March 2025
    cal = calendar.monthcalendar(year, month)
    return cal, bd_holidays

@app.route("/")
def index():
    return render_template("index.html", companies=companies)

@app.route("/forecast", methods=["POST"])
def forecast():
    company = request.form["company"]
    predictions_7 = predict_stock_prices(company, 7)
    predictions_15 = predict_stock_prices(company, 15)
    predictions_30 = predict_stock_prices(company, 30)
    save_plot(predictions_7, "forecast_7.png", 7)
    save_plot(predictions_15, "forecast_15.png", 15)
    save_plot(predictions_30, "forecast_30.png", 30)
    return jsonify({"success": True})

if __name__ == "__main__":
    if not os.path.exists("static/plots"):
        os.makedirs("static/plots")
    app.run(debug=True)
