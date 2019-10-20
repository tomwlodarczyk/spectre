#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
from bs4 import BeautifulSoup
import pybtex.database
import re


def append_eprint_links_to_citelist(html_dir, references_file):
    logger = logging.getLogger(__name__)

    # Load `citelist.html` file as structured HTML
    with open(os.path.join(html_dir, 'citelist.html'), 'r') as citelist_file:
        citelist = BeautifulSoup(citelist_file, 'html.parser')
    # Load `References.bib` as structured bibliography database
    references = pybtex.database.parse_file(references_file)

    for citeref_anchor in citelist.find_all(id=re.compile('CITEREF_*')):
        key = re.match('CITEREF_(.*)', citeref_anchor.attrs['id']).groups()[0]
        entry = references.entries[key]
        # We support only arXiv eprints for now. More formats may be added here
        # if necessary.
        if 'archivePrefix' in entry.fields \
                and entry.fields['archivePrefix'] == 'arXiv' \
                and 'eprint' in entry.fields:
            logger.debug("Found arXiv link for {}.".format(key))
            arxiv_url = "https://arxiv.org/abs/{}".format(
                entry.fields['eprint'])
            eprint_link = citelist.new_tag('a', href=arxiv_url)
            eprint_link.string = "arXiv:{}".format(entry.fields['eprint'])
        else:
            logger.debug("Found no supported eprint data for {}.".format(key))
            continue
        # Append the eprint link to the citation HTML
        citeref_item = citeref_anchor.find_next('p')
        citeref_item.append(eprint_link)
        citeref_item.append('.')
        logger.info("Added eprint link to {}: {}".format(key, eprint_link))

    # Write the `citelist.html` file back to disk
    with open(os.path.join(html_dir, 'citelist.html'), 'w') as citelist_file:
        citelist_file.write(str(citelist))


def parse_args():
    """
    Parse the command line arguments
    """
    import argparse as ap
    parser = ap.ArgumentParser(
        description="""
        Postprocess the documentation generated by Doxygen. Currently only adds
        eprint links to the bibliography.
        """)
    parser.add_argument(
        '--html-dir',
        required=True,
        help="""
        Doxygen's HTML output directory. Files in this directory will be
        modified by the postprocessing.
        """)
    parser.add_argument(
        '--references-file',
        required=True,
        help="""
        The .bib file that Doxygen used to generate the bibliography.
        """)
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0,
        help="Verbosity (-v, -vv, ...)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Set the log level
    logging.basicConfig(level=logging.WARNING - args.verbose * 10)

    logger = logging.getLogger(__name__)
    logger.info("Postprocessing docs HTML directory: {}".format(args.html_dir))

    logger.info("Appending eprint links to citelist...")
    append_eprint_links_to_citelist(args.html_dir, args.references_file)

    logger.info("Done postprocessing docs.")
