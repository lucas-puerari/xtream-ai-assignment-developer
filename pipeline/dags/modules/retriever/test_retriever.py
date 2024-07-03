import pytest
import requests
from requests.exceptions import HTTPError
from retriever import download_data


@pytest.fixture(name="mock_requests_get_success")
def mock_requests_get_success_fixture(mocker):
    mock_get = mocker.patch.object(requests, 'get')
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b'Test content'
    return mock_get


@pytest.fixture(name="mock_requests_get_failure")
def mock_requests_get_failure_fixture(mocker):
    # pylint: disable=unused-argument
    def mock_get(url):
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = HTTPError("Mocked HTTP Error")
        return mock_response

    mocker.patch.object(requests, 'get', side_effect=mock_get)

def test_download_success(mock_requests_get_success):
    url = 'https://example.com/file.csv'
    result = download_data(url)
    assert isinstance(result, str)
    assert len(result) > 0


def test_download_failure(mock_requests_get_failure):
    url = 'https://example.com/non_existent_file.csv'
    with pytest.raises(HTTPError):
        download_data(url)
