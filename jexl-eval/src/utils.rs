use regex::Regex;

lazy_static! {
    /// This is an example for using doc comment attributes
    static ref VERSION_RE: Regex = Regex::new(
        r"(?x)
^[\s]*(?P<major>[0-9]+)  # major
[.]
(?P<minor>[0-9]+)  # minor
[.]
(?P<patch>[0-9]+)  # patch
").unwrap();

}

pub (crate) fn parse_semver(text: &str) -> anyhow::Result<semver::Version> {
    if let Some(matches) = VERSION_RE.captures(&text) {
        let major = matches.name("major").unwrap().as_str().parse::<u64>()?;
        let minor = matches.name("minor").unwrap().as_str().parse::<u64>()?;
        let patch = matches.name("patch").unwrap().as_str().parse::<u64>()?;

        Ok(semver::Version {
            major,
            minor,
            patch,
            pre: Default::default(),
            build: Default::default(),
        })
    } else {
        Err(anyhow::anyhow!("not a valid version string"))
    }
}