from jobpilot.tools.linkedin_jd import format_job_description, parse_linkedin_job


def test_parse_linkedin_public_job_markup() -> None:
    html = """
    <html>
      <body>
        <main>
          <section class="top-card-layout">
            <h1 class="top-card-layout__title">Senior AI Engineer</h1>
            <a class="topcard__org-name-link">Relevance AI</a>
          </section>
          <section class="description">
            <h2>About the job</h2>
            <div class="description__text">
              <div class="show-more-less-html__markup">
                <p>We are seeking Senior Engineers to join our AI team.</p>
                <ul>
                  <li>Develop Agents and Multi-agent systems.</li>
                  <li>Evaluate and improve AI performance using tests.</li>
                </ul>
                <button>Show more</button>
              </div>
            </div>
          </section>
        </main>
      </body>
    </html>
    """

    job = parse_linkedin_job(html)

    assert job.title == "Senior AI Engineer"
    assert job.company == "Relevance AI"
    assert "We are seeking Senior Engineers" in job.about_job
    assert "Develop Agents and Multi-agent systems." in job.about_job
    assert "Show more" not in job.about_job


def test_format_job_description_includes_expected_sections() -> None:
    job = parse_linkedin_job(
        """
        <h1>Senior Software Engineer</h1>
        <a class="topcard__org-name-link">Relevance AI</a>
        <section class="description">
          <h2>About the job</h2>
          <div class="show-more-less-html__markup">Build reliable APIs.</div>
        </section>
        """
    )

    text = format_job_description(job)

    assert text == (
        "Title: Senior Software Engineer\n"
        "Company: Relevance AI\n\n"
        "About the job:\n"
        "Build reliable APIs.\n"
    )
