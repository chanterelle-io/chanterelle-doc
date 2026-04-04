import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const benefits = [
  {
    icon: '🔒',
    title: 'Local & Secure',
    desc: "Runs entirely on your machine. No data leaves your environment before you're ready.",
  },
  {
    icon: '📁',
    title: 'Code-Friendly',
    desc: 'JSON + Python project files you can version in Git and review like source code.',
  },
  {
    icon: '⚡',
    title: 'Zero Infrastructure',
    desc: "No servers or deployments. Open a project file and you're ready to go.",
  },
  {
    icon: '🤖',
    title: 'AI-Assisted',
    desc: 'Built-in skill for GitHub Copilot, Cursor, and Windsurf to generate valid project files from natural language.',
  },
];

function BenefitCard({icon, title, desc}: {icon: string; title: string; desc: string}) {
  return (
    <div className={styles.benefitCard}>
      <span className={styles.benefitIcon}>{icon}</span>
      <div>
        <strong className={styles.benefitTitle}>{title}</strong>
        <p className={styles.benefitDesc}>{desc}</p>
      </div>
    </div>
  );
}

function HomepageHeader() {
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.heroLogoTitle}>
          <img
            src="img/chanterelle.png"
            alt="Chanterelle Logo"
            className={styles.heroLogo}
            loading="eager"
            decoding="async"
          />
          <Heading as="h1" className={styles.heroTitle}>Chanterelle</Heading>
        </div>
        <p className={styles.heroTagline}>The bridge between Jupyter and production</p>
        <p className={styles.heroSubtitle}>
          A lightweight desktop app for data scientists and ML engineers to{' '}
          <strong>test, present, and share</strong> models and findings — without deploying anything.
        </p>
        <div className={styles.heroCtas}>
          <Link className="button button--primary button--lg" to="/docs/intro">
            Get Started
          </Link>
          <Link
            className={clsx('button button--outline button--lg', styles.outlineBtn)}
            to="/docs/install"
          >
            Download
          </Link>
        </div>
        <div className={styles.heroBadges}>
          <span className={styles.badge}>v1.0</span>
          <span className={styles.badge}>Python 3.8+</span>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="The bridge between Jupyter and production — a lightweight desktop app for data scientists and ML engineers.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />

        <section className={styles.whySection}>
          <div className="container">
            <Heading as="h2" className={styles.sectionHeading}>Why Chanterelle?</Heading>
            <div className={styles.benefitsGrid}>
              {benefits.map((b, i) => (
                <BenefitCard key={i} {...b} />
              ))}
            </div>
          </div>
        </section>

        <section className={styles.videoSection}>
          <div className="container">
            <Heading as="h2" className={styles.sectionHeading}>See It in Action</Heading>
            <div className={styles.videoWrapper}>
              <iframe
                src="https://www.youtube.com/embed/coK2rDEzb8o?si=J1d5UP9HjvkTwdXk"
                title="Chanterelle Demo"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                referrerPolicy="strict-origin-when-cross-origin"
                allowFullScreen
              />
            </div>
          </div>
        </section>

        <section className={styles.ctaSection}>
          <div className="container">
            <Heading as="h2" className={styles.ctaHeading}>Ready to get started?</Heading>
            <p className={styles.ctaSubtext}>
              Download Chanterelle and set up your first project in minutes.
            </p>
            <div className={styles.heroCtas}>
              <Link className="button button--primary button--lg" to="/docs/intro">
                Read the Docs
              </Link>
              <Link
                className={clsx('button button--outline button--lg', styles.outlineBtn)}
                to="/docs/install"
              >
                Download
              </Link>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
