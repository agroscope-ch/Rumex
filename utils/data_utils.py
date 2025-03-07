from darwin.client import Client


def get_dataset_version_from_darwin(dataset_name, team = 'digital-production'):
    '''A function to get the latest version of a dataset from Darwin
    Args:
        team: The team name
        data: A dictionary containing the dataset name
    Returns:
        version_id: The latest version of the dataset
    '''

    client = Client.local()
    dataset = client.get_remote_dataset(team + '/' +  dataset_name)
    latest_release = dataset.get_release()

    version_id = latest_release.name
    return version_id
