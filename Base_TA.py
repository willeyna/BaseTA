import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fmin_l_bfgs_b
import photospline as psp

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# MISC UTILITY FUNCTIONS ---
#used in kent dist; spherical dot product
def sph_dot(th1,th2,phi1,phi2):
    return np.sin(th1)*np.sin(th2)*np.cos(phi1-phi2) + np.cos(th1)*np.cos(th2)

def logit(x):
    return np.log(x/(1-x))

def inv_logit(x):
    return 1/(1+np.exp(-x))

def dinv_logit(x):
    return np.exp(x)/(np.exp(x)+1)**2

def GreatCircleDistance(ra_1, dec_1, ra_2, dec_2):
        '''Compute the great circle distance between two events'''
        '''All coordinates must be given in radians'''
        delta_dec = np.abs(dec_1 - dec_2)
        delta_ra = np.abs(ra_1 - ra_2)
        x = (np.sin(delta_dec / 2.))**2. + np.cos(dec_1) *\
            np.cos(dec_2) * (np.sin(delta_ra / 2.))**2.
        return 2. * np.arcsin(np.sqrt(x))

def evPSFd(nue,numu):
    kappa = 1./(nue[2])**2
    log_dist = np.log(kappa) - np.log(2*np.pi) - kappa + kappa*sph_dot(np.pi/2-nue[1], np.pi/2-numu[1], nue[0], numu[0])
    return np.exp(log_dist)

####################################################################################

class tester():

    '''
    methods: [list] Consist of method names as strings (Written exactly as they appear in package)
    tracks: [int] # background tracks
    cascades: [int] # background cascades
    resolution: [int] Healpy grid resolution (NPIX = 2**resolution)

    args: [dict] Used to pass information to the methods. Keys are method specific strings (ex: 'Prior' to pass in a TC prior).
                Values vary depending on method-- use this dict to pass any arguments to methods. Can be accessed by indexing the tester object as a dict.
                List of args can be found in the README

    Takes in the above arguments, checks to make sure they're of the right form, then creates pdfs from MC and initializes the object
    '''
    def __init__(self, tracks, cascades, resolution = 8, args = dict()):

        self.args = args

        #radius of circle around source
        if 'delta_ang' not in args:
            #default area to consider possibly signal
            args['delta_ang'] = np.deg2rad(20)

        self.track_count = tracks
        self.cascade_count = cascades

        self.load_pdfs()

        return

    #Pulls using ow and can inject events. [Rob]
    def gen(self, n_Ev, g, topo = 0, inra=None, indec=None):
            if(g<=0):
                print("g (second arg) must be >0, negative sign for spectra is hard-coded")
                return
            if topo == 0:
                mc = np.load("./mcdata/tracks_mc.npy")
            elif topo == 1:
                mc = np.load("./mcdata/cascade_mc.npy")
            else:
                print("topo = 0 for tracks, topo = 1 for cascades")
                return

            if self['NORTH']:
                mc = mc[mc['sinDec'] > 0]

            p=mc["ow"]*np.power(mc['trueE'],-g)
            p/=np.sum(p)
            keySC=np.random.choice( np.arange(len(p)), n_Ev, p=p, replace=False)
            evs=np.copy(mc[keySC])

            if(inra!=None and indec!=None):
                #Note: this method was yanked from a skylab example and might not actually be great
                eta = np.random.uniform(0., 2.*np.pi, n_Ev)
                sigmags=np.random.normal(scale=evs["angErr"])

                evs["dec"] = indec + np.sin(eta) * sigmags
                evs["ra"] = inra + np.cos(eta) * sigmags

                changeDecs=evs['dec']> np.pi/2
                #over shooting in dec is the same as rotating arounf and subtracting the Dec from pi.
                evs['ra'][changeDecs]+=np.pi #rotate the point to the other side
                evs['dec'][changeDecs]=np.pi-evs['dec'][changeDecs] #move the Dec accordingly

                #undershooting in dec
                changeDecs=evs['dec']< -np.pi/2

                evs['ra'][changeDecs]+=np.pi #rotate the point to the other side
                evs['dec'][changeDecs]=-np.pi-evs['dec'][changeDecs] #move the Dec accordingly

                #under or overshooting in ra, a bit easier
                evs['ra'][evs['ra']>2*np.pi]-=2*np.pi
                evs['ra'][evs['ra']<0]+=2*np.pi

            # 100 GeV ENERGY CUT FOR CURRENT SPLINES
            evs = evs[evs['logE'] >= 2]
            return evs

    #cuts the sky for events around source and minimizes llh, returning TS
    def analyze(self, tracks, cascades, src_ra, src_dec):
        # cuts for tracks within delta_ang of the source-- does NOT cut on cascade events
        if tracks is not None:
            source_tracks = tracks[GreatCircleDistance(tracks['ra'], tracks['dec'], src_ra, src_dec) < self['delta_ang']]

        #PATH IF TRACKS OR CASCADES MISSING (1 sample)
        samples = [tracks, cascades]
        if None in samples:
            samples.pop(samples.index(None))
            B = self.fB(samples[0], onesamp = True)
            N = samples[0].shape[0]
            x,llh,warn = fmin_l_bfgs_b(self.llh, x0 = (10,2.5), bounds = ((0,1000),(1,4)), fprime = None, approx_grad = False, args = (samples[0], None, src_ra, src_dec, B, N))
            TS = -2*llh
            return TS, x, warn

        evs = np.concatenate([source_tracks, cascades])
        #number of events considered in the signal part of llh

        #total event count
        N = tracks.shape[0] + cascades.shape[0]

        track_B = self.fB(source_tracks)
        casc_B  = self.fB(cascades)
        B = np.concatenate([track_B, casc_B])

        x,llh,warn = fmin_l_bfgs_b(self.llh, x0 = (10,2.5), bounds = ((0,1000),(1,4)), fprime = None, approx_grad = False, args = (source_tracks, cascades, src_ra, src_dec, B, N))

        #negative max llh
        maxllh = -llh

        TS = 2*(maxllh)
        return TS, x, warn

    #not built to be called on its own-- rather through analyze which passes in the right args
    def llh(self, x, tracks, cascades, src_ra, src_dec, B, N):
        # x = (n, gamma)
        ns = x[0]
        gamma = x[1]

        #PATH IF TRACKS OR CASCADES MISSING (1 sample)
        if cascades is None:
            deltaN = N - (tracks.shape[0])
            S = self.f_psi(tracks, src_ra, src_dec, gamma)*self.f_energy(tracks, src_ra, src_dec, gamma)* self.f_tau(tracks, src_ra, src_dec, gamma)
            llh_vals = (ns/N)*(S/B - 1) + 1
            logllh = np.sum(np.log(llh_vals)) + deltaN*np.log(1-(ns/N))
            #gradient calculation
            dl_dns = (np.sum((S/B - 1.)/llh_vals) - (deltaN)/(1.-(ns/N)))/N
            #when only one sample is present we drop p(tau)
            product_rule = (self.f_psi(tracks, src_ra, src_dec, gamma, dgamma = True)*self.f_energy(tracks, src_ra, src_dec, gamma) +
                              self.f_psi(tracks, src_ra, src_dec, gamma)*self.f_energy(tracks, src_ra, src_dec, gamma, dgamma = True))
            dl_dgamma = np.sum((ns/N)/(llh_vals * B) * product_rule)
            return (-logllh, (-dl_dns, -dl_dgamma))

        deltaN = N - (tracks.shape[0] + cascades.shape[0])

        track_S = self.f_psi(tracks, src_ra, src_dec, gamma)*self.f_energy(tracks, src_ra, src_dec, gamma)* self.f_tau(tracks, src_ra, src_dec, gamma)
        casc_S = self.f_psi(cascades, src_ra, src_dec, gamma)*self.f_energy(cascades, src_ra, src_dec, gamma)* self.f_tau(cascades, src_ra, src_dec, gamma)
        S = np.concatenate([track_S, casc_S])

        llh_vals = (ns/N)*(S/B - 1) + 1

        logllh = np.sum(np.log(llh_vals)) + deltaN*np.log(1-(ns/N))

        #gradient calculation
        dl_dns = (np.sum((S/B - 1.)/llh_vals) - (deltaN)/(1.-(ns/N)))/N

        product_rule_tracks = (self.f_psi(tracks, src_ra, src_dec, gamma, dgamma = True)*self.f_energy(tracks, src_ra, src_dec, gamma)* self.f_tau(tracks, src_ra, src_dec, gamma) +
                              self.f_psi(tracks, src_ra, src_dec, gamma)*self.f_energy(tracks, src_ra, src_dec, gamma, dgamma = True)* self.f_tau(tracks, src_ra, src_dec, gamma) +
                              self.f_psi(tracks, src_ra, src_dec, gamma)*self.f_energy(tracks, src_ra, src_dec, gamma)* self.f_tau(tracks, src_ra, src_dec, gamma, dgamma = True))

        product_rule_cascades = (self.f_psi(cascades, src_ra, src_dec, gamma, dgamma = True)*self.f_energy(cascades, src_ra, src_dec, gamma)* self.f_tau(cascades, src_ra, src_dec, gamma) +
                              self.f_psi(cascades, src_ra, src_dec, gamma)*self.f_energy(cascades, src_ra, src_dec, gamma, dgamma = True)* self.f_tau(cascades, src_ra, src_dec, gamma) +
                              self.f_psi(cascades, src_ra, src_dec, gamma)*self.f_energy(cascades, src_ra, src_dec, gamma)* self.f_tau(cascades, src_ra, src_dec, gamma, dgamma = True))
        product_rule = np.concatenate([product_rule_tracks, product_rule_cascades])

        dl_dgamma = np.sum((ns/N)/(llh_vals * B) * product_rule)

        return (-logllh, (-dl_dns, -dl_dgamma))

    def fB(self, events, onesamp = False):
        topo = events['topo'][0]
        # .97,.03 represent the topology ratio for background events-- get this from data eventually
        if not topo:
            BT = self['BT'].evaluate_simple([events['logE'], events['sinDec']])
            if not onesamp:
                return BT * .97
            return BT

        else:
            BC = np.exp(self['BC'].evaluate_simple([events['sinDec'], events['logE']])) / (2*np.pi)
            if not onesamp:
                return BC * (1-0.97)
            return BC

    #functions to evaluate the llh signal observable functions based on topology
    #dgamma decision trees because of inconsistency in observable placement in spline calls
    def f_psi(self, events, src_ra, src_dec, gamma=2, dgamma=False):
        topo = events['topo'][0]
        psi = GreatCircleDistance(events['ra'], events['dec'], src_ra, src_dec)
        if not topo:
            if not dgamma:
                return self['ST'].evaluate_simple([np.log10(events['angErr']), events['logE'], np.log10(psi), np.full(events.shape[0], gamma)])/(psi *np.log(10) * np.sin(psi))
            else:
                return self['ST'].evaluate_simple([np.log10(events['angErr']), events['logE'], np.log10(psi), np.full(events.shape[0], gamma)], 8)/(psi *np.log(10) * np.sin(psi))
        else:
            #cascade spatial term
            if not dgamma:
                return evPSFd([events['ra'],events['dec'],events['angErr']], [src_ra, src_dec])
            else:
                #for dumb cascade spatial pdfs that don't change with gamma
                return 0

    def f_energy(self, events, src_ra, src_dec, gamma=2, dgamma=False):
        topo = events['topo'][0]
        if not topo:
            if not dgamma:
                return self['ET'].evaluate_simple([events['logE'], np.full(events.shape[0], np.sin(src_dec)), np.full(events.shape[0], gamma)])
            else:
                return self['ET'].evaluate_simple([events['logE'], np.full(events.shape[0], np.sin(src_dec)), np.full(events.shape[0], gamma)], 4)
        else:
            if not dgamma:
                return np.exp(self['EC'].evaluate_simple([np.full(events.shape[0], gamma), np.full(events.shape[0], np.sin(src_dec)), events['logE']]))
            else:
                return (self['EC'].evaluate_simple([np.full(events.shape[0], gamma), np.full(events.shape[0], np.sin(src_dec)), events['logE']],1) *
                        np.exp(self['EC'].evaluate_simple([np.full(events.shape[0], gamma), np.full(events.shape[0], np.sin(src_dec)), events['logE']])))

    def f_tau(self, events, src_ra, src_dec, gamma=2, dgamma=False):
        topo = events['topo'][0]
        ptau = inv_logit(self['Tau'].evaluate_simple([np.full(events.shape[0], gamma), np.sin(src_dec)]))
        if not topo:
            if not dgamma:
                return 1 - ptau
            else:
                return -(self['Tau'].evaluate_simple([np.full(events.shape[0], gamma), np.sin(src_dec)], 1) * dinv_logit(ptau))
        else:
            if not dgamma:
                return ptau
            else:
                return self['Tau'].evaluate_simple([np.full(events.shape[0], gamma), np.sin(src_dec)], 1) * dinv_logit(ptau)

    #allow the tester to be indexed directly rather than having to call tester.args
    #args acts as a container for the testers' objects
    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, val):
        self.args[key] = val

    def __repr__(self):
        return f'Multi-tester object crafted for the Topology Aware LLH method.\nBackground tracks: {self.track_count} Background Cascades: {self.cascade_count}\nWill be stored in the file "{self.pkl}" with tester name "{self.name}"'


    '''
    Ran during initialization of a tester object to store pdfs in tester.args
    Loads in MC data to create Background Spatial pdfs and Energy pdfs for signal and background (-3.7 spectrum)
    Creates pdfs for tracks and cascades separately for use in topology-implemented methods
    '''
    def load_pdfs(self):

        track_spatial = psp.SplineTable('./splines/sig_E_psi_photospline_v006_4D.fits')
        track_energy = psp.SplineTable('./splines/E_dec_photospline_v006_3D.fits')
        track_bkg = psp.SplineTable('./splines/bg_2d_photospline.fits')

        #casc_spatial = psp.SplineTable('./splines/sig_E_psi_photospline_v006_4D.fits')
        casc_energy = psp.SplineTable('./splines/cascade_E_dec_photospline_v000_3D.fits')
        casc_bkg = psp.SplineTable('./splines/cascade_bg_2d_photospline.fits')

        #temporary northern sky only ptau made with Northern Tracks and DNN Cascades
        topology = psp.SplineTable('./splines/tau_photospline_v000_2D.fits')

        #fills in tester with splines for energy, background spatial term, and topology for both split topology and non split topology searches
        self.args['ST'] = track_spatial
        self.args['SC'] = None

        self.args['BT'] = track_bkg
        self.args['BC'] = casc_bkg

        self.args['ET'] = track_energy
        self.args['EC'] = casc_energy

        self.args['Tau'] = topology

    #wrapper function for creating a number of events and calculating TS
    def test_methods(self, ra, dec, ninj_t = 0, ninj_c = 0, gamma = 2, return_fit = False):
        tracks = np.concatenate([self.gen(ninj_t, gamma, 0, inra = ra, indec = dec),self.gen(self.track_count, 3.7, 0)])
        cascades = np.concatenate([self.gen(ninj_c, gamma, 1, inra = ra, indec = dec),self.gen(self.cascade_count, 3.7, 1)])

        if return_fit:
            return self.analyze(tracks, cascades, ra, dec)[0], self.analyze(tracks, cascades, ra, dec)[1]
        return self.analyze(tracks, cascades, ra, dec)[0]
