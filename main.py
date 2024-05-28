import pyomo.environ as pyo

# モデルの作成
model = pyo.ConcreteModel(name="Stock Optimization", doc="Optimization of stock indicators from 2010 to 2023")

# 変数の定義 (ROE、PBR、PER、DIV、MV、MOM の重み)
num_indicators = 6
model.w = pyo.Var(range(1, num_indicators + 1), bounds=(0, 1), initialize=0.5)

# 指標データの定義（2010年から2023年まで）
indicators = [
    [0.028569900293578000, 0.029169621030612300, 0.038634212836734700, 0.058941791694444400, 0.004716123761467910, 0.026023502289719700],
    [-0.033859569219434000, -0.108917004247423000, -0.125681543804124000, -0.023862959229245300, -0.188232716084906000, -0.050660310697169800],
    [0.183257393695238000, 0.171120840378947000, 0.164006775873684000, 0.240675382825714000, 0.217476172952381000, 0.182905222923810000],
    [0.548508780837255000, 0.545502150301076000, 0.637857369247312000, 0.521989789215686000, 0.537788421131373000, 0.716824581078431000],
    [0.151685386475247000, 0.151581039029565000, 0.173957586203478000, 0.144985207263168000, 0.115981337891089000, 0.096178932784313700],
    [0.144553797340000000, 0.081764319681318700, 0.087807143798351600, 0.131862775160000000, 0.107830596233400000, 0.121097040960000000],
    [0.034716837612244900, 0.060462390629213400, 0.111436685235955000, 0.059859906112244900, 0.003080071612244990, -0.034813505408163300],
    [0.276458377268041000, 0.225468275286250000, 0.350787305068182000, 0.276067940680412000, 0.215908908948454000, 0.319686178061856000],
    [-0.208513565625000000, -0.167220801465517000, -0.197904335431035000, -0.140970948239583000, -0.143898870968750000, -0.307261882395833000],
    [0.223685309313830000, 0.228404933995081000, 0.234604113367174000, 0.123583787617021000, 0.173008690707447000, 0.156956025591245000],
    [0.123170903118279000, -0.059502250428235300, -0.057795450905882300, -0.096568429477419400, 0.102332428053763000, 0.161247105905376000],
    [0.046568918141304300, 0.196973690437036000, 0.230825561964655000, 0.269572626304348000, 0.137809116771739000, 0.017638768203380500],
    [-0.024060451296666700, 0.199541286682927000, 0.171929564024390000, 0.173264438333333000, 0.000967422981111188, 0.015853893355555600],
    [0.294335058865169000, 0.360656835102605000, 0.254898689781617000, 0.377252953685393000, 0.305017088101124000, 0.283137639643944000]
]

# 目的関数の定義
model.OBJ = pyo.Objective(
    expr=sum(
        sum(indicators[i][j] * model.w[j + 1] for j in range(num_indicators))
        for i in range(len(indicators))
    ),
    sense=pyo.maximize
)

# 制約条件の定義 (各指標の重みの合計が1)
model.Constraint = pyo.Constraint(expr=sum(model.w[i] for i in range(1, num_indicators + 1)) == 1)

# 最適化ソルバを設定
opt = pyo.SolverFactory('ipopt')

# 最適化計算を実行
res = opt.solve(model)

# 結果の表示
optimal_weights = [model.w[i].value for i in range(1, num_indicators + 1)]
optimal_value = model.OBJ()

indicators_names = ['ROE', 'PBR', 'PER', 'DIV', 'MV', 'MOM']

print(f"評価関数：{optimal_value}")
cumulative_contribution = 0.0
for i in range(num_indicators):
    weight_percentage = optimal_weights[i] * 100
    cumulative_contribution += weight_percentage
    print(f"{indicators_names[i]}: Weight = {optimal_weights[i]:.40f} ({weight_percentage:.5f}%), Cumulative Contribution = {cumulative_contribution:.5f}%")
