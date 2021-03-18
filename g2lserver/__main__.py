#!/usr/bin/env python3

import connexion

from g2lserver import encoder
import os

def main():

    app = connexion.App(__name__, specification_dir='./swagger/')
    app.app.json_encoder = encoder.JSONEncoder
    app.add_api('swagger.yaml', arguments={'title': 'Planning Data Science Workflows from a grammar'})
    app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
