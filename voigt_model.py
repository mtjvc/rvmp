#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as pl
from pyspeckit.spectrum.models.inherited_voigtfitter import voigt

from spectrum import Spectrum
from gma.utils import create_rave_filename

import george
from george import kernels
import emcee
import triangle


CaT_lines = [8498.02, 8542.09, 8662.14]


def lnlike(p, x, y, yerr):
    a, alpha, b, beta, c, zeta = np.exp(p[:6])
    gp = george.GP(a * kernels.ExpSquaredKernel(alpha) +
                   b * kernels.ExpSquaredKernel(beta) +
                   c * kernels.ExpSquaredKernel(zeta))

    lnl = 0.0
    for i in range(3):
        pos = CaT_lines[i] + p[6]
        amp = p[7 + i * 3]
        sigma = p[8 + i * 3]
        gamma = p[9 + i * 3]

        try:
            gp.compute(x[i], yerr[i])
        except np.linalg.linalg.LinAlgError, ValueError:
            return -np.inf
        v = voigt(x[i], amp, pos, sigma, gamma)
        if np.inf in abs(v):
            lnl = -np.inf
        else:
            lnl += gp.lnlikelihood(y[i] + v - 1, quiet=True)

    return lnl


def lnprior(p):
    lna, lnalpha, lnb, lnbeta, lnc, lnzeta, xcen,\
        amp1, sigma1, gamma1, amp2, sigma2, gamma2, amp3, sigma3, gamma3 = p
    # Gaussian priors center at different wiggle lengths
    lnp = 0.0
    lnp += -(lnalpha + 0.5) ** 2 / (2 * 0.6 ** 2)
    lnp += -(lnbeta - 1.25) ** 2 / (2 * 0.8 ** 2)
    lnp += -(lnzeta - 3.0) ** 2 / (2 * 0.6 ** 2)

    if (-50. < lna < 0. and -50. < lnb < 0. and -50. < lnc < 0.
        and -5. < xcen < 5.
        and amp1 > 0. and amp2 > 0. and amp3 > 0.
        and sigma1 > 0. and sigma2 > 0. and sigma3 > 0.
        and gamma1 > 0. and gamma2 > 0. and gamma3 > 0.):
        return lnp
    return -np.inf


def lnprob(p, x, y, yerr, nwalkers):
    global itercount

    sys.stdout.write("\rIteration: %.0f" % (itercount / float(nwalkers)))
    sys.stdout.flush()
    itercount += 1

    lp = lnprior(p)
    return lp + lnlike(p, x, y, yerr) if np.isfinite(lp) else -np.inf


def run(raveid, snr, wlwidth=12, nwalkers=128, initial=None, preruniter=10,
        finaliter=10, calc_snr=False):
    global itercount

    a = raveid.split('_')
    fn = create_rave_filename(a[0], a[1], int(a[2]))

    try:
        len(snr)
    except TypeError:
        snr = [snr, snr, snr]

    # generic initial conditions if not given
    if initial is None:
        initial = np.array([-9.0, -0.5, -8.0, 1.25, -8.0, 3.0, 0.0, 0.6, 0.3,
                            0.6, 1.1, 0.4, 0.7, 1.1, 0.3, 0.7])

    try:
        spec = Spectrum(fn)
    except IOError:
        return

    sel = [[], []]
    for i in range(3):
        sel[0].append(spec.x[(spec.x > CaT_lines[i] - wlwidth) &
                             (spec.x < CaT_lines[i] + wlwidth)])
        sel[1].append(spec.y[(spec.x > CaT_lines[i] - wlwidth) &
                             (spec.x < CaT_lines[i] + wlwidth)])

    x, y, yerr = sel[0], sel[1],\
        [i[1] / i[0] for i in zip(snr, np.ones_like(sel[0]))]

    ndim = len(initial)
    p0 = np.array([np.array(initial) + 1e-2 * np.random.randn(ndim)
                   for i in xrange(nwalkers)])
    p0[:, 0] = np.random.rand(nwalkers) * 30 - 35
    p0[:, 2] = np.random.rand(nwalkers) * 30 - 35
    p0[:, 4] = np.random.rand(nwalkers) * 30 - 35

    data = [x, y, yerr, nwalkers]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    itercount = 0
    p0, lnp, _ = sampler.run_mcmc(p0, preruniter)
    sampler.reset()
    print '\n First run done.'

    itercount = 0
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, preruniter)
    sampler.reset()
    print '\n Second run done.'

    itercount = 0
    p0, _, _ = sampler.run_mcmc(p0, finaliter)
    print '\n Finished.'

    samples = sampler.flatchain

    # Calculate new SNR values
    if calc_snr:
        nsnr = []
        for i in range(3):
            gpmodels = []
            for s in samples[np.random.randint(len(samples), size=100)]:
                pos = CaT_lines[i] + s[6]
                amp = s[7 + i * 3]
                sigma = s[8 + i * 3]
                gamma = s[9 + i * 3]

                # Use only the second and the third kernel
                gp = george.GP(
                    # np.exp(s[0]) * kernels.ExpSquaredKernel(np.exp(s[1])) +
                    np.exp(s[2]) * kernels.ExpSquaredKernel(np.exp(s[3])) +
                    np.exp(s[4]) * kernels.ExpSquaredKernel(np.exp(s[5])))
                gp.compute(x[i], yerr[i])

                m1 = 1 - voigt(x[i], amp, pos, sigma, gamma)
                m2 = gp.sample_conditional(y[i] - 1 +
                                           voigt(x[i], amp, pos, sigma, gamma),
                                           x[i]) + m1
                gpmodels.append(m2)

            gpa = np.array(gpmodels).T
            gpavg = np.average(gpa, axis=1)

            # New SNR is 1 / sigma of the residuals
            nsnr.append(1.0 / np.std(y[i] - gpavg))

        return samples, sampler.flatlnprobability, np.array(nsnr)

    return samples, sampler.flatlnprobability


def plot_samples(raveid, snr, samples, lnproblty, wlwidth=12, size=10,
                 ysh=[0., 0., 0.]):
    a = raveid.split('_')
    fn = create_rave_filename(a[0], a[1], int(a[2]))

    try:
        len(snr)
    except TypeError:
        snr = [snr, snr, snr]

    try:
        spec = Spectrum(fn)
    except IOError:
        return

    sel = [[], []]
    for i in range(3):
        sel[0].append(spec.x[(spec.x > CaT_lines[i] - wlwidth) &
                             (spec.x < CaT_lines[i] + wlwidth)])
        sel[1].append(spec.y[(spec.x > CaT_lines[i] - wlwidth) &
                             (spec.x < CaT_lines[i] + wlwidth)])

    x, y, yerr = sel[0], sel[1],\
        [i[1] / i[0] for i in zip(snr, np.ones_like(sel[0]))]

    pl.figure()

    for i in range(3):
        gpmodels = []
        linemodels = []
        ew = []
        xs = np.linspace(CaT_lines[i] - wlwidth, CaT_lines[i] + wlwidth, 200)
        for s in samples[np.random.randint(len(samples), size=size)]:
            pos = CaT_lines[i] + s[6]
            amp = s[7 + i * 3]
            sigma = s[8 + i * 3]
            gamma = s[9 + i * 3]

            gp = george.GP(
                np.exp(s[0]) * kernels.ExpSquaredKernel(np.exp(s[1])) +
                np.exp(s[2]) * kernels.ExpSquaredKernel(np.exp(s[3])) +
                np.exp(s[4]) * kernels.ExpSquaredKernel(np.exp(s[5])))
            gp.compute(x[i], yerr[i])

            m1 = 1 - voigt(xs, amp, pos, sigma, gamma)
            m2 = gp.sample_conditional(y[i] - 1 +
                                       voigt(x[i], amp, pos, sigma, gamma),
                                       xs) + m1
            linemodels.append(m1)
            gpmodels.append(m2)
            ew.append(np.sum(1 - m1[1:]) * (xs[1:] - xs[:-1]))

        pl.errorbar(x[i] - CaT_lines[i], y[i] + i * ysh[i], yerr=yerr[i],
                    fmt=".k", capsize=0)
        pl.text(-12., 1.07 + i * ysh[i], '%.2f +- %.2f A' %
                (np.average(ew), np.std(ew)))

        la = np.array(linemodels).T
        lstd = np.std(la, axis=1)
        lavg = np.average(la, axis=1)
        y1, y2 = lavg + lstd + i * ysh[i], lavg - lstd + i * ysh[i]
        pl.fill_between(xs - CaT_lines[i], y1, y2, alpha=0.3)

        gpa = np.array(gpmodels).T
        gpstd = np.std(gpa, axis=1)
        gpavg = np.average(gpa, axis=1)
        y1, y2 = gpavg + gpstd + i * ysh[i], gpavg - gpstd + i * ysh[i]
        pl.fill_between(xs - CaT_lines[i], y1, y2, color='r', alpha=0.3)

    pl.ylim(0.5, 2.0)

    # pl.figure(figsize=(14, 3))
    # for sc, s in enumerate(samples.T):
    #     print s
    #     ax = pl.subplot(2, 8, sc + 1)
    #     pl.hist(s[::30], bins=50, histtype='step')
    #     for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #         label.set_fontsize(6)
    # label.set_fontsize(13)

    samples = np.array([np.array(i[:6]) for i in
                        samples[np.random.randint(len(samples),
                                size=len(samples) / 20)]])
    triangle.corner(samples)

    pl.show()


def pipeline(raveid, snr):

    nwalkers = 128
    # Prerun (mostly to get a better SNR estimate)
    samples, lnproblty, snr = run(raveid, snr, nwalkers=nwalkers,
                                  preruniter=10, finaliter=10,
                                  calc_snr=True)

    # Use best sample from the prerun as a better intial estimate
    initial = samples[np.argmax(lnproblty)]

    f = open('%s.1.npy' % raveid, 'wb')
    np.save(f, np.array([samples, lnproblty, snr]))
    f.close()

    plot_samples(raveid, snr, samples, lnproblty, size=100,
                 ysh=[0.0, 0.45, 0.4])

    # Production run
    samples, lnproblty, snr = run(raveid, snr, nwalkers=nwalkers,
                                  initial=initial, preruniter=10,
                                  finaliter=10, calc_snr=True)

    f = open('%s.2.npy' % raveid, 'wb')
    np.save(f, np.array([samples, lnproblty, snr]))
    f.close()

    plot_samples(raveid, snr, samples, lnproblty, size=100,
                 ysh=[0.0, 0.45, 0.4])


if __name__ == '__main__':
    # raveid = '20121217_0356m54_109'
    raveid = '20040409_1200m15_140'
    snr = 26.0

    pipeline(raveid, snr)

    # if 0:
    #     samples, lnproblty, snr = run(raveid, snr, preruniter=20,
    #                                   finaliter=50, calc_snr=True)
    #     f = open('%s.1.npy' % raveid, 'wb')
    #     np.save(f, np.array([samples, lnproblty, snr]))
    #     f.close()
    # else:
    #     samples, lnproblty, snr = np.load('%s.1.npy' % raveid)
    #
    # print snr
    # plot_samples(raveid, snr, samples, lnproblty,
    #              size=20, ysh=[0.0, 0.45, 0.4])
