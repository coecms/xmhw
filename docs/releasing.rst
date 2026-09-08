==================
Publishing to PyPI
==================

PyPI releases are published from GitHub Releases with OpenID Connect. No PyPI
API token needs to be stored in GitHub.

One-time setup
==============

Before the first release, an owner of the ``coecms`` GitHub organization and a
PyPI account owner must:

#. Create a ``pypi`` environment under the GitHub repository's
   ``Settings > Environments`` page. Add deployment protection rules if the
   organization requires an approval before publication.
#. On PyPI's ``Publishing`` page, add a pending publisher with these values:

   * PyPI project name: ``xmhw``
   * GitHub owner: ``coecms``
   * GitHub repository: ``xmhw``
   * Workflow filename: ``publish.yml``
   * Environment name: ``pypi``

The pending publisher creates the PyPI project when the workflow publishes the
first release.

Publishing a release
====================

#. Confirm the test workflow passes on the commit to release.
#. Create a GitHub release with a new PEP 440-compatible tag, such as ``1.0.1``.
   PBR derives the distribution version from this Git tag.
#. Publish the GitHub release. The ``Publish to PyPI`` workflow builds the wheel
   and source archive, validates both with ``twine check --strict``, and
   publishes them using the trusted publisher.
#. Confirm the release appears at https://pypi.org/project/xmhw/ and test it in a
   clean environment with ``python -m pip install xmhw``.

PyPI does not permit replacing an existing distribution. If publication fails
after uploading either archive, create a new patch release rather than reusing
the tag or version.
