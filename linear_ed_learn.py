#Solves NK model with learning about change in intercept of rule.

#compile model
#what is the ordering of parameters in p0 -- where do I find in yaml file?
from MarkovSwitchingModel import MarkovSwitchingDSGE
nklearn = MarkovSwitchingDSGE.read('3eq_ms.yaml')
nk = nklearn.compile_model()
p0 = nklearn.p0()
nkirf = nk.impulse_response(p0,h=1)


