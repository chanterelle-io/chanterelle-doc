import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroTitleDiv}>
          <img 
            src="img/chanterelle.png" 
            alt="Chanterelle Logo" 
            className={styles.heroLogo}
            loading="eager"
            decoding="async"
          />
          <span className={clsx('chanterelle', styles.heroTitle)}>
            {siteConfig.title}
          </span>
        </div>
        <p className="hero__subtitle chanterelle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="The Chanterelle documentation site.">
      <HomepageHeader />
      <main>
        <div className="container" style={{textAlign: 'center', padding: '2rem 0'}}>
          <iframe width="80%" height="500" src="https://www.youtube.com/embed/coK2rDEzb8o?si=J1d5UP9HjvkTwdXk" title="YouTube video player" frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerPolicy="strict-origin-when-cross-origin" allowFullScreen></iframe>
        </div>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
