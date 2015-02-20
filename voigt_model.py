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


# Wavelengts sourced from NIST ASD database
CaT_lines = [8498.02, 8542.09, 8662.14]


def lnlike(p, x, y, yerr, usegp):
    if usegp:
        # Use GPs to model the noise
        st = 6
        a, alpha, b, beta, c, zeta = np.exp(p[:6])

        # Three ExpSquared kernels to model three different wiggle scales
        gp = george.GP(a * kernels.ExpSquaredKernel(alpha) +
                       b * kernels.ExpSquaredKernel(beta) +
                       c * kernels.ExpSquaredKernel(zeta))
    else:
        st = 0

    lnl = 0.0
    for i in range(3):
        pos = CaT_lines[i] + p[st]
        amp = p[st + 1 + i * 3]
        sigma = p[st + 2 + i * 3]
        gamma = p[st + 3 + i * 3]

        if usegp:
            try:
                gp.compute(x[i], yerr[i])
            except np.linalg.linalg.LinAlgError, ValueError:
                return -np.inf
        v = voigt(x[i], amp, pos, sigma, gamma)
        if np.inf in abs(v):
            lnl = -np.inf
        else:
            if usegp:
                lnl += gp.lnlikelihood(y[i] + v - 1, quiet=True)
            else:
                lnl += -0.5 * (np.sum(((y[i] + v - 1) /
                               yerr[i]) ** 2))
    return lnl


def lnprior(p):
    lna, lnalpha, lnb, lnbeta, lnc, lnzeta, xcen,\
        amp1, sigma1, gamma1, amp2, sigma2, gamma2, amp3, sigma3, gamma3 = p

    # Gaussian priors centered at different wiggle scales
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


def lnprior_wogp(p):
    xcen, amp1, sigma1, gamma1, amp2, sigma2, gamma2, \
        amp3, sigma3, gamma3 = p

    if (-5. < xcen < 5.
        and amp1 > 0. and amp2 > 0. and amp3 > 0.
        and sigma1 > 0. and sigma2 > 0. and sigma3 > 0.
        and gamma1 > 0. and gamma2 > 0. and gamma3 > 0.):
        return 0.0

    return -np.inf


def lnprob(p, x, y, yerr, nwalkers, usegp):
    if usegp:
        lp = lnprior(p)
    else:
        lp = lnprior_wogp(p)
    return lp + lnlike(p, x, y, yerr, usegp) if np.isfinite(lp) else -np.inf


def run(raveid, snr, wlwidth=12, nwalkers=128, initial=None, preruniter=10,
        finaliter=10, calc_snr=False, usegp=True):

    a = raveid.split('_')
    fn = create_rave_filename(a[0], a[1], int(a[2]))

    # Use a single value for initial SNR estimate and use different value
    # for each line once there is a better estimate
    try:
        len(snr)
    except TypeError:
        snr = [snr, snr, snr]

    # generic initial conditions if not given
    if initial is None:
        if usegp:
            st = 6
            initial = np.array([-9.0, -0.5, -8.0, 1.25, -8.0, 3.0, 0.0, 0.6,
                                0.3, 0.6, 1.1, 0.4, 0.7, 1.1, 0.3, 0.7])
        else:
            st = 0
            initial = np.array([0.0, 0.6, 0.3, 0.6, 1.1, 0.4, 0.7, 1.1, 0.3,
                                0.7])

    try:
        spec = Spectrum(fn)
    except IOError:
        return

    # Select only wlwidth wide range left and right from each line
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

    # Box-randomize amplitudes to speed-up the convergence
    if usegp:
        p0[:, 0] = np.random.rand(nwalkers) * 30 - 35
        p0[:, 2] = np.random.rand(nwalkers) * 30 - 35
        p0[:, 4] = np.random.rand(nwalkers) * 30 - 35

    data = [x, y, yerr, nwalkers, usegp]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print 'Starting %s...' % raveid

    p0, lnp, _ = sampler.run_mcmc(p0, preruniter)
    sampler.reset()
    print 'First run done.'

    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-2 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, preruniter)
    sampler.reset()
    print 'Second run done.'

    p0, _, _ = sampler.run_mcmc(p0, finaliter)
    print 'Finished %s.' % raveid

    samples = sampler.flatchain

    # Calculate new SNR values
    if calc_snr:
        nsnr = []
        for i in range(3):
            models = []
            for s in samples[np.random.randint(len(samples), size=100)]:
                pos = CaT_lines[i] + s[st]
                amp = s[st + 1 + i * 3]
                sigma = s[st + 2 + i * 3]
                gamma = s[st + 3 + i * 3]

                m1 = 1 - voigt(x[i], amp, pos, sigma, gamma)

                if usegp:
                    # Use only the second and the third kernel
                    gp = george.GP(
                        # np.exp(s[0]) *
                        #kernels.ExpSquaredKernel(np.exp(s[1])) +
                        np.exp(s[2]) * kernels.ExpSquaredKernel(np.exp(s[3])) +
                        np.exp(s[4]) * kernels.ExpSquaredKernel(np.exp(s[5])))
                    gp.compute(x[i], yerr[i])

                    m2 = gp.sample_conditional(y[i] - 1 +
                                               voigt(x[i], amp, pos,
                                                     sigma, gamma),
                                               x[i]) + m1
                    models.append(m2)
                else:
                    models.append(m1)

            ma = np.array(models).T
            mavg = np.average(ma, axis=1)

            # New SNR is 1 / sigma of the residuals
            nsnr.append(1.0 / np.std(y[i] - mavg))

        return samples, sampler.flatlnprobability, np.array(nsnr)

    return samples, sampler.flatlnprobability


def plot_samples(raveid, snr, samples, lnproblty, nwalkers, outdir, nr,
                 wlwidth=12, size=10, ysh=[0., 0., 0.], usegp=True):
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

    fig = pl.figure()

    ews = []
    for i in range(3):
        if usegp:
            gpmodels = []
            st = 6
        else:
            st = 0
        linemodels = []
        ew = []
        xs = np.linspace(CaT_lines[i] - wlwidth, CaT_lines[i] + wlwidth, 200)
        for s in samples[np.random.randint(len(samples), size=size)]:
            pos = CaT_lines[i] + s[st]
            amp = s[st + 1 + i * 3]
            sigma = s[st + 2 + i * 3]
            gamma = s[st + 3 + i * 3]

            m1 = 1 - voigt(xs, amp, pos, sigma, gamma)
            linemodels.append(m1)

            if usegp:
                gp = george.GP(
                    np.exp(s[0]) * kernels.ExpSquaredKernel(np.exp(s[1])) +
                    np.exp(s[2]) * kernels.ExpSquaredKernel(np.exp(s[3])) +
                    np.exp(s[4]) * kernels.ExpSquaredKernel(np.exp(s[5])))
                gp.compute(x[i], yerr[i])

                m2 = gp.sample_conditional(y[i] - 1 +
                                           voigt(x[i], amp, pos, sigma, gamma),
                                           xs) + m1
                gpmodels.append(m2)
            ew.append(np.sum(1 - m1[1:]) * (xs[1:] - xs[:-1]))

        ews.append((np.average(ew), np.std(ew)))
        pl.errorbar(x[i] - CaT_lines[i], y[i] + i * ysh[i], yerr=yerr[i],
                    fmt=".k", capsize=0)
        pl.text(-12., 1.07 + i * ysh[i], '%.2f +- %.2f A' %
                (np.average(ew), np.std(ew)))

        la = np.array(linemodels).T
        lstd = np.std(la, axis=1)
        lavg = np.average(la, axis=1)
        y1, y2 = lavg + lstd + i * ysh[i], lavg - lstd + i * ysh[i]
        pl.fill_between(xs - CaT_lines[i], y1, y2, alpha=0.3)

        if usegp:
            gpa = np.array(gpmodels).T
            gpstd = np.std(gpa, axis=1)
            gpavg = np.average(gpa, axis=1)
            y1, y2 = gpavg + gpstd + i * ysh[i], gpavg - gpstd + i * ysh[i]
            pl.fill_between(xs - CaT_lines[i], y1, y2, color='r', alpha=0.3)

    pl.ylim(0.5, 2.0)
    pl.savefig(outdir + '%s.%d.png' % (raveid, nr))
    fig.clf()
    pl.close(fig)

    fig = pl.figure()
    chns = lnproblty.reshape((nwalkers, len(samples) / nwalkers))
    for c in chns:
        pl.plot(c, 'k', alpha=0.1, lw=3)
    med = [np.median(i) for i in chns.T]
    pl.plot(med, 'r')
    pl.savefig(outdir + '%s.%d.chain.png' % (raveid, nr))
    fig.clf()
    pl.close(fig)

    if nr == 2 and usegp:
        plotsamples = np.array([np.array(i[:6]) for i in
                                samples[np.random.randint(len(samples),
                                        size=len(samples) / 20)]])
        triangle.corner(plotsamples)
        pl.savefig(outdir + '%s.%d.tri.kernelpars.png' % (raveid, nr))
        pl.clf()

        for j in range(3):
            plotsamples = np.array([np.array(i[7 + j:10 + j]) for i in
                                    samples[np.random.randint(len(samples),
                                            size=len(samples) / 20)]])
            triangle.corner(plotsamples)
            pl.savefig(outdir + '%s.%d.tri.linepars.%d.png' % (raveid, nr, j))
            pl.clf()

    return ews


def pipeline(raveid, snr, outdir):
    # Only works with GP samples

    nwalkers = 128
    # Prerun (mostly to get a better SNR estimate)
    samples, lnproblty, snr = run(raveid, snr, nwalkers=nwalkers,
                                  preruniter=200, finaliter=500,
                                  calc_snr=True)

    # Use best sample from the prerun as a better initial estimate
    initial = samples[np.argmax(lnproblty)]

    f = open(outdir + '%s.1.npy' % raveid, 'wb')
    np.save(f, np.array([samples, lnproblty, snr]))
    f.close()

    plot_samples(raveid, snr, samples, lnproblty, nwalkers, outdir, 1,
                 size=100, ysh=[0.0, 0.45, 0.4])

    # Production run
    samples, lnproblty, snr = run(raveid, snr, nwalkers=nwalkers,
                                  initial=initial, preruniter=1000,
                                  finaliter=5000, calc_snr=True)

    f = open(outdir + '%s.2.npy' % raveid, 'wb')
    np.save(f, np.array([samples, lnproblty, snr]))
    f.close()

    plot_samples(raveid, snr, samples, lnproblty, nwalkers, outdir, 2,
                 size=100, ysh=[0.0, 0.45, 0.4])


def pipeline_worker(p):
    pipeline(*p)


if __name__ == '__main__':
    test_gp = False
    raveid = '20060521_1742m83_109'
    snr = 60.0
    outdir = 'trash_metal/test/'

    if test_gp:
        samples, lnproblty = run(raveid, snr, nwalkers=128, preruniter=500,
                                 finaliter=500)
        plot_samples(raveid, snr, samples, lnproblty, 128, outdir, 2,
                     ysh=[0.0, 0.45, 0.4])
    else:
        samples, lnproblty = run(raveid, snr, nwalkers=128, preruniter=200,
                                 finaliter=500, usegp=False)
        ews = plot_samples(raveid, snr, samples, lnproblty, 128, outdir, 2,
                           ysh=[0.0, 0.45, 0.4], usegp=False, size=100)
